import sys
import argparse
import yaml
import math
import random
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor

from .knn_monitor import knn_predict
from tqdm import tqdm

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ActCLR_Processor(PT_Processor):
    """
        Processor for AimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        
        if epoch <= 100:
            loader.dataset.aug_method = random.sample(['1', '2', '3', '4', '5'], 5)
            loader.dataset.strong_aug_method = random.sample(['1', '2', '3', '4', '5', '8', '9'], 7)
        elif epoch > 100 and epoch <= 200:
            loader.dataset.aug_method = random.sample(['1', '2', '3', '4', '5'], 3)
            loader.dataset.strong_aug_method = random.sample(['1', '2', '3', '4', '5', '8', '9'], 5)
        elif epoch > 200 and epoch <= 300:
            loader.dataset.aug_method = random.sample(['1', '2', '3', '4', '5'], 1)
            loader.dataset.strong_aug_method = random.sample(['1', '2', '3', '4', '5', '8', '9'], 3)
        elif epoch > 300:
            loader.dataset.temperal_padding_ratio = -1
            loader.dataset.shear_amplitude = -1
            loader.dataset.aug_method = ''
            loader.dataset.strong_aug_method = ''
        
        loss_value = []

        for data1, data2, data3, data4 in tqdm(loader):
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            data4 = data4.float().to(self.dev, non_blocking=True)    


            data1 = self.view_gen(data1)
            data2 = self.view_gen(data2)
            data3 = self.view_gen(data3)
            data4 = self.view_gen(data4)

            # forward
            loss = self.model(data1, data2, data3, data4, epoch=epoch)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            if self.local_rank == 0:
                self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
        if self.local_rank == 0:
            self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    def view_gen(self, data):
        if self.arg.stream == 'joint':
            pass
        elif self.arg.stream == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif self.arg.stream == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            bone = torch.zeros_like(data)

            for v1, v2 in Bone:
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

            data = bone
        else:
            raise ValueError

        return data
    
    @torch.no_grad()
    def knn_monitor(self, epoch):
        self.model.module.encoder_q.eval()
        feature_bank, label_bank = [], []
        with torch.no_grad():
            # generate feature bank
            for data, label in tqdm(self.data_loader['mem_train'], desc='Feature extracting'):
                data = data.float().to(self.dev, non_blocking=True)
                label = label.long().to(self.dev, non_blocking=True)

                data = self.view_gen(data)
                [feature] = self.model.module.encoder_q(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                label_bank.append(label)
            # [D, N]
            feature_bank = concat_all_gather(torch.cat(feature_bank, dim=0)).t().contiguous()
            # [N]
            feature_labels = concat_all_gather(torch.cat(label_bank)).to(feature_bank.device)
            # loop test data to predict the label by weighted knn search
            for i in self.arg.knn_k:
                total_top1, total_top5, total_num = 0, 0, 0
                test_bar = tqdm(self.data_loader['mem_test'], desc='kNN-{}'.format(i))
                for data, label in test_bar:
                    data = data.float().to(self.dev, non_blocking=True)
                    label = label.float().to(self.dev, non_blocking=True)

                    data = self.view_gen(data)

                    [feature] = self.model.module.encoder_q(data)
                    feature = concat_all_gather(F.normalize(feature, dim=1))

                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += feature.size(0)
                    total_top1 += (pred_labels[:, 0] == concat_all_gather(label)).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100

                self.knn_results[i][epoch] = acc
                if self.local_rank == 0:
                    self.train_writer.add_scalar('KNN-{}'.format(i), acc, epoch)

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        return parser

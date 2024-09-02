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


class AimCLR_Processor(PT_Processor):
    """
        Processor for AimCLR Pre-training.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for [data1, data2, data3], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            data3 = data3.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            data1 = self.view_gen(data1)
            data2 = self.view_gen(data2)
            data3 = self.view_gen(data3)

            # forward
            if epoch <= self.arg.mining_epoch:
                output1, target1, output2, output3, target2 = self.model(data1, data2, data3)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = self.loss(output1, target1)
                loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                loss3 = -torch.mean(torch.sum(torch.log(output3) * target2, dim=1))  # DDM loss
                loss = loss1 + (loss2 + loss3) / 2.
            else:
                output1, mask, output2, output3, target2 = self.model(data1, data2, data3, nnm=True, topk=self.arg.topk)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output1.size(0))
                else:
                    self.model.update_ptr(output1.size(0))
                loss1 = - (F.log_softmax(output1, dim=1) * mask).sum(1) / mask.sum(1)
                loss1 = loss1.mean()
                loss2 = -torch.mean(torch.sum(torch.log(output2) * target2, dim=1))  # DDM loss
                loss3 = -torch.mean(torch.sum(torch.log(output3) * target2, dim=1))  # DDM loss
                loss = loss1 + (loss2 + loss3) / 2.

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
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss'] = np.mean(loss_value)
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

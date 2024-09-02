import sys
import argparse
import yaml
import math
import numpy as np
from time import time

# torch
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

total_sample_num = 400
plot_class = [7, 8, 13, 14, 42, 50, 25, 26]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Visualize_Processor(Processor):
    """
        Processor for Linear Evaluation.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

        for name, param in self.model.encoder_q.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        self.num_grad_layers = 2
        if hasattr(self.model, 'encoder_q_motion'):
            for name, param in self.model.encoder_q_motion.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            self.num_grad_layers += 2
        if hasattr(self.model, 'encoder_q_bone'):
            for name, param in self.model.encoder_q_bone.named_parameters():
                if name not in ['fc.weight', 'fc.bias']:
                    param.requires_grad = False
            self.num_grad_layers += 2

        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(parameters) == self.num_grad_layers
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    # def train(self, epoch):
    #     self.model.eval()
    #     self.adjust_lr()
    #     loader = self.data_loader['train']
    #     loss_value = []

    #     for data, label in loader:
    #         self.global_step += 1
    #         # get data
    #         data = data.float().to(self.dev, non_blocking=True)
    #         label = label.long().to(self.dev, non_blocking=True)
            
    #         if self.arg.stream == 'joint':
    #             pass
    #         elif self.arg.stream == 'motion':
    #             motion = torch.zeros_like(data)

    #             motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

    #             data = motion
    #         elif self.arg.stream == 'bone':
    #             Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
    #                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
    #                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

    #             bone = torch.zeros_like(data)

    #             for v1, v2 in Bone:
    #                 bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    
    #             data = bone
    #         else:
    #             raise ValueError

    #         # forward
    #         output = self.model(None, data)
    #         loss = self.loss(output, label)

    #         # backward
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()

    #         # statistics
    #         self.iter_info['loss'] = loss.data.item()
    #         self.iter_info['lr'] = '{:.6f}'.format(self.lr)
    #         loss_value.append(self.iter_info['loss'])
    #         self.show_iter_info()
    #         self.meta_info['iter'] += 1
    #         self.train_log_writer(epoch)

    #     self.epoch_info['train_mean_loss']= np.mean(loss_value)
    #     self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
    #     self.show_epoch_info()


    def RGB_to_Hex(self,rgb):
        RGB = rgb#(0,0,0)            # 将RGB格式划分开来
        color = '#'
        for i in RGB:
            num = int(i)
            # 将R、G、B分别转化为16进制拼接转换并大写  hex() 函数用于将10进制整数转换成16进制，以字符串形式表示
            color += str(hex(num))[-2:].replace('x', '0').upper()
        return color

    def plot_embedding(self, data, label, title):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure(figsize=(7,5))
        #plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        ax = plt.subplot(111)
        
        col = [(141,211,199),(255,255,179),(190,186,218),(251,128,114),(128,177,211),(253,180,98),(179,222,105),(252,205,229),(217,217,217),(188,128,189),(204,235,197),
(255,237,111)]

        col = [(166,206,227),(31,120,180),(178,223,138),(51,160,44),(251,154,153),(227,26,28),(253,191,111),(255,127,0),(202,178,214),(106,61,154),(255,255,153),(177,89,40),]
        for i in range(len(col)):
            col[i] = self.RGB_to_Hex(col[i])

        #col = ['#FFD700','#0000CD','#FF4500','#FF1493','#698B69','#8470FF','#00FF7F','#A0522D','#A020F0','#D02090']
        #annot = ['sky','ridge','soil','sand','bedrock','rock','rover','trace','hole']
        num_class = 12
        #col = np.linspace(0.0,1.0,num=num_class)

        f = [True,]*num_class
        to_be_plotted = plot_class
        print('total sample num:', data.shape[0])
        for i in range(data.shape[0]):
            #plt.text(data[i, 0], data[i, 1], str(label[i]),
             #       color=plt.cm.Set1(label[i] / 10.),
              #      fontdict={'weight': 'bold', 'size': 9})
            if label[i] not in to_be_plotted:
                continue
            s = 15
            first = False
            for j in plot_class:
                j_id = plot_class.index(j)
                #print(j_id)
                if label[i]==j:
                    plt.scatter(data[i,0],data[i,1],s=s,c=col[j_id],alpha=1.0,linewidth=0)              
        plt.title(title)
        return fig

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        predict_frag = []
        label_frag = []
        count = 0
        for data, label in loader:
            count += 1
            if count > total_sample_num:
                break
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            
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
                    
                data = bone
            else:
                raise ValueError

            # inference
            with torch.no_grad():
                output, predict = self.model(None, data)
            result_frag.append(output.data.cpu().numpy())
            predict_frag.append(predict.data.cpu().numpy())

            # get loss
            # loss = self.loss(output, label)
            # loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.predict = np.concatenate(predict_frag)
        self.label = np.concatenate(label_frag)

        # self.eval_info['eval_mean_loss']= np.mean(loss_value)
        self.show_eval_info()

        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)

        t0 = time()
        result = tsne.fit_transform(self.result)

        fig = self.plot_embedding(result, self.label,
                            'Feature embedding(time %.2fs)'
                            % (time() - t0))

        plt.savefig('aimclr_joint.pdf')

        # show top-k accuracy 
        # for k in self.arg.show_topk:
        #     self.show_topk(k)
        # self.show_best(1)

        # self.eval_log_writer(epoch)

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.print_networks(self.model)
        self.test(0)

    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
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

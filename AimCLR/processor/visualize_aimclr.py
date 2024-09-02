import sys
import argparse
import yaml
import math
import random
import numpy as np
from tqdm import tqdm
from time import time
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

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

from .processor import Processor
from .pretrain import PT_Processor

total_sample_num = 400
plot_class = [51, 42, 41, 39,  8, 54, 18, 34,  7, 14, 44, 53]#, 15, 43, 36, 46, 40, 24, 38]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Visualize_Processor(Processor):
    """
        Processor for AimCLR Pre-training.
    """
    def __init__(self, argv=None):
        super().__init__(argv)
    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()
    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            if self.meta_info['epoch'] < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (self.meta_info['epoch']) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        if k==1:
            self.test_acc = int(accuracy*100000)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

        if k==1:
            acc_per_class = []
            num_per_class = []
            for i in range(60):
                idx = np.zeros_like(self.label)
                idx [(self.label == i)] = 1.0
                acc_per_class.append(sum(np.array(hit_top_k)*np.array(idx))/sum(idx))
                num_per_class.append(sum(idx))
            print(np.array(acc_per_class).argsort())
            print(np.array(num_per_class)[np.array(acc_per_class).argsort()])
            

    
    def get_data(self, epoch=0):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        data_x = []
        data_y = []
        process = tqdm(loader)
        count = 0
        for data, label in process:
            count+=1
            if count>total_sample_num:
                break
            # get data
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # inference
            with torch.no_grad():
                feat,output = self.model(None,data,return_feat=True)
            result_frag.append(output.data.cpu().numpy())
            if count==1:
                data_x = feat.data.cpu().numpy()
                data_y = label.data.cpu().numpy()
            else:
                data_x = np.concatenate([data_x,feat.data.cpu().numpy()],axis=0)
                data_y = np.concatenate([data_y,label.data.cpu().numpy()],axis=0)

            label_frag.append(label.data.cpu().numpy())


        self.result = np.concatenate(result_frag)
        
        if 1:
            self.label = np.concatenate(label_frag)

            # show top-k accuracy
            for k in self.arg.show_topk:
                self.show_topk(k)
        return np.array(data_x),np.array(data_y)
    
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

    def tsne_func(self):
        import pickle
        if self.arg.ensemble:
            joint_path = '/mnt/netdisk/zhangjh/Code/AimCLR-main/work_dir/NTU60/xview/joint_skeletonclr/linear_tmp/'
            bone_path = '/mnt/netdisk/zhangjh/Code/AimCLR-main/work_dir/NTU60/xview/bone_skeletonclr/linear_tmp/'
            motion_path = '/mnt/netdisk/zhangjh/Code/AimCLR-main/work_dir/NTU60/xview/motion_skeletonclr/linear_tmp/'
            r1 = open(joint_path + 'test_result.pkl', 'rb')
            r1 = list(pickle.load(r1).items())
            joint_score = np.array(list(np.array(r1)[:,1]))

            r2 = open(bone_path + 'test_result.pkl', 'rb')
            r2 = list(pickle.load(r2).items())
            bone_score = np.array(list(np.array(r2)[:,1]))

            r3 = open(motion_path + 'test_result.pkl', 'rb')
            r3 = list(pickle.load(r3).items())
            motion_score = np.array(list(np.array(r3)[:,1]))
            
            d = (joint_score+bone_score+motion_score)/3.0
            print(d.shape)
            l = open('/mnt/netdisk/linlilang/CrosSCLR/data/NTU-RGB-D/xview/val_label.pkl', 'rb')
            l = np.array(pickle.load(l)[1])
            print(l.shape)

        else:
            d, l= self.get_data(0)
        idx = []
        for i in range(l.shape[0]):
            if l[i] in plot_class:
                idx.append(i)
        data = d[idx]
        label = l[idx]
        #data, label= lbp_data(opts, model, loader, device)
        print('Computing t-SNE embedding')
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        t0 = time()
        result = tsne.fit_transform(data)
        fig = self.plot_embedding(result, label,
                            'Feature embedding(time %.2fs)'
                            % (time() - t0))
        plt.savefig('feature.pdf')
    #plt.show(fig)
    
    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        self.print_networks(self.model)
        self.tsne_func()
        

    # @staticmethod
    # def get_parser(add_help=False):

    #     # parameter priority: command line > config > default
    #     parent_parser = Processor.get_parser(add_help=False)
    #     parser = argparse.ArgumentParser(
    #         add_help=add_help,
    #         parents=[parent_parser],
    #         description='Spatial Temporal Graph Convolution Network')

    #     # region arguments yapf: disable
    #     # evaluation
    #     parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
    #     parser.add_argument('--warm_up_epoch', type=int, default=0, help='which Top K accuracy will be shown')
    #     # optim
    #     parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
    #     parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
    #     parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    #     parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
    #     parser.add_argument('--ensemble', type=str2bool, default=False, help='use nesterov or not')
    #     parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
    #     # endregion yapf: enable

    #     return parser

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        parser.add_argument('--warm_up_epoch', type=int, default=0, help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--ensemble', type=str2bool, default=False, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        return parser


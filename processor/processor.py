import sys
import argparse
import yaml
import os
import shutil
import numpy as np
import random
import math

# torch
import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
from tensorboardX import SummaryWriter

import subprocess

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)

        self.global_step = 0

    def train_log_writer(self, epoch):
        self.train_writer.add_scalar('batch_loss', self.iter_info['loss'], self.global_step)
        self.train_writer.add_scalar('lr', self.lr, self.global_step)
        self.train_writer.add_scalar('epoch', epoch, self.global_step)

    def eval_log_writer(self, epoch):
        self.val_writer.add_scalar('eval_loss', self.eval_info['eval_mean_loss'], epoch)
        self.val_writer.add_scalar('current_result', self.current_result, epoch)
        self.val_writer.add_scalar('best_result', self.best_result, epoch)

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.eval_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        dist.init_process_group(backend='nccl')

    def load_optimizer(self):
        pass

    def load_data(self):
        self.data_loader = dict()

        if self.arg.train_feeder_args:
            train_feeder = import_class(self.arg.train_feeder)
            train_dataset = train_feeder(**self.arg.train_feeder_args)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True,
                sampler=self.train_sampler)

        if self.arg.test_feeder_args:
            test_feeder = import_class(self.arg.test_feeder)
            test_dataset = test_feeder(**self.arg.test_feeder_args)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True,
                sampler=self.test_sampler)

        if self.arg.mem_train_feeder_args:
            mem_train_feeder = import_class(self.arg.mem_train_feeder)
            mem_train_dataset = mem_train_feeder(**self.arg.mem_train_feeder_args)
            self.mem_train_sampler = torch.utils.data.distributed.DistributedSampler(mem_train_dataset)
            self.data_loader['mem_train'] = torch.utils.data.DataLoader(
                dataset=mem_train_dataset,
                batch_size=self.arg.batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False,
                sampler=self.mem_train_sampler)

        if self.arg.mem_test_feeder_args:
            mem_test_feeder = import_class(self.arg.mem_test_feeder)
            mem_test_dataset = mem_test_feeder(**self.arg.mem_test_feeder_args)
            self.mem_test_sampler = torch.utils.data.distributed.DistributedSampler(mem_test_dataset)
            self.data_loader['mem_test'] = torch.utils.data.DataLoader(
                dataset=mem_test_dataset,
                batch_size=self.arg.test_batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False,
                sampler=self.mem_test_sampler)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_eval_info(self):
        for k, v in self.eval_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('eval', self.meta_info['iter'], self.eval_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['train_mean_loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.eval_info['test_mean_loss'] = 1
        self.show_eval_info()

    def print_networks(self, net, print_flag=False):
        self.io.print_log('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if print_flag:
            self.io.print_log(net)
        self.io.print_log('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        self.io.print_log('-----------------------------------------------')

    def start(self, local_rank):
        if local_rank == 0:
            # get the output of `git diff`
            diff_output = subprocess.check_output(['git', 'diff'])
            # print the output
            self.io.print_log(diff_output.decode('utf-8'))
        
        
        self.local_rank = local_rank
        torch.cuda.set_device(local_rank)
        self.device = torch.cuda.device(local_rank)
        self.dev = local_rank
        self.gpus = list(range(torch.distributed.get_world_size()))
        
        if self.arg.phase == 'train' and local_rank == 0:
            if os.path.isdir(self.arg.work_dir + '/train'):
                print('log_dir: ', self.arg.work_dir, 'already exist')
                shutil.rmtree(self.arg.work_dir + '/train')
                shutil.rmtree(self.arg.work_dir + '/val')
                print('Dir removed: ', self.arg.work_dir + '/train')
                print('Dir removed: ', self.arg.work_dir + '/val')
            self.train_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'train'), 'train')
            self.val_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'val'), 'val')
        # else:
        #     self.train_writer = self.val_writer = SummaryWriter(os.path.join(self.arg.work_dir, 'test'), 'test')
            
        self.load_model()
        self.load_weights()
        # self.gpu()
        self.load_data()
        self.load_optimizer()
            
        if local_rank == 0:    
            self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.print_networks(self.model)

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        self.model = self.model.cuda()
        
        self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )

        # training phase
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            self.meta_info['iter'] = self.global_step
            self.best_result = 0.0
            self.knn_results = dict()
            self.KNN_epoch_results = dict()
            for k in self.arg.knn_k:
                self.knn_results[k] = dict()
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train_sampler.set_epoch(epoch)
                self.meta_info['epoch'] = epoch + 1

                # training
                if local_rank == 0:   
                    self.io.print_log('Training epoch: {}'.format(epoch + 1))
                self.train(epoch + 1)

                # save model
                if self.arg.save_interval == -1:
                    pass
                elif (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and dist.get_rank() == 0:
                    filename = 'epoch{}_model.pt'.format(epoch + 1)
                    self.io.save_model(self.model, filename)

                # evaluation
                if self.arg.eval_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    if local_rank == 0:   
                        self.io.print_log('Eval epoch: {}'.format(epoch + 1))
                    self.test(epoch + 1)
                    self.io.print_log("current %.2f%%, best %.2f%%" % 
                                            (self.current_result, self.best_result))

                    # save best model
                    filename = 'epoch%.3d_acc%.2f_model.pt' % (epoch + 1, self.current_result)
                    self.io.save_model(self.model, filename)
                    if self.current_result >= self.best_result:
                        filename = 'best_model.pt'
                        self.io.save_model(self.model, filename)
                        # save the output of model
                        if self.arg.save_result:
                            result_dict = dict(
                                zip(self.data_loader['test'].dataset.sample_name,
                                    self.result))
                            self.io.save_pkl(result_dict, 'test_result.pkl')

                if self.arg.knn_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.knn_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    if local_rank == 0:   
                        self.io.print_log('KNN eval epoch {}'.format(epoch + 1))
                    self.knn_monitor(epoch + 1)

                    for k in self.arg.knn_k:
                        self.io.print_log("KNN - {} current: {:.2f}%, best: {:.2f}%".format(
                            k, self.knn_results[k][epoch + 1], max(self.knn_results[k].values())
                            ))
                        if epoch + 1 in self.arg.KNN_show:
                            if epoch + 1 not in self.KNN_epoch_results:
                                self.KNN_epoch_results[epoch + 1] = dict()
                            self.KNN_epoch_results[epoch + 1][k] = [self.knn_results[k][epoch + 1], max(self.knn_results[k].values())]

                    if (epoch + 1 == self.arg.num_epoch):
                        self.io.print_log('*' * 10 + ' KNN Result ' + '*' * 10)
                        for show_epoch in self.arg.KNN_show:
                            if show_epoch in self.KNN_epoch_results:
                                for k in self.arg.knn_k:
                                    self.io.print_log('\t{}-{}: cur-{:.2f}%, best-{:.2f}%'.format(
                                        show_epoch, k, self.KNN_epoch_results[show_epoch][k][0],
                                        self.KNN_epoch_results[show_epoch][k][1]))

        # test phase
        elif self.arg.phase == 'test':
            
            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            self.best_result = 0.0

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test(1)
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result_%.3d.pkl'% (epoch + 1))

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=True,
                            help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#epoch)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#epoch)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        parser.add_argument('--knn_interval', type=int, default=5, help='the interval for knn models (#epoch)')

        # feeder
        parser.add_argument('--train_feeder', default='feeder.feeder', help='train data loader will be used')
        parser.add_argument('--test_feeder', default='feeder.feeder', help='test data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--mem_train_feeder', default='feeder.feeder',
                            help='memory train data loader will be used for knn')
        parser.add_argument('--mem_test_feeder', default='feeder.feeder',
                            help='memory test data loader will be used for knn')
        parser.add_argument('--mem_train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--mem_test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')

        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--rename_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
                            
        parser.add_argument('--knn_k', type=int, default=[], nargs='+', help='KNN-K')
        parser.add_argument('--knn_classes', type=int, default=60, help='use cosine lr schedule')
        parser.add_argument('--knn_t', type=float, default=0.1, help='use cosine lr schedule')
        parser.add_argument('--KNN_show', type=int, default=[], nargs='+',
                            help='the epoch to show the best KNN result')
        # endregion yapf: enable

        return parser

import random
import numpy as np
import pickle, torch
from . import tools


class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        return data_numpy


class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for triple inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True,
                 aug_method='12345', strong_aug_method='1234589'):
        self.data_path = data_path
        self.label_path = label_path
        self.aug_method = aug_method
        self.strong_aug_method = strong_aug_method

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])

        # processing
        data1 = self._strong_aug(data_numpy)
        data2 = self._aug(data_numpy)
        data3 = self._weak_aug(data_numpy)
        data4 = self._weak_aug(data_numpy)
        return data1, data2, data3, data4
    
    def _weak_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        return data_numpy

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.aug_method:
            data_numpy = tools.gaus_noise(data_numpy)
        if '4' in self.aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.aug_method:
            data_numpy = tools.random_time_flip(data_numpy)
        if '7' in self.aug_method:
            data_numpy = tools.dropout(data_numpy)
        if '8' in self.aug_method:
            data_numpy = tools.interpolate(data_numpy, 1)
        if '9' in self.aug_method:
            data_numpy = tools.rescale(data_numpy, 0.5)
            
        return data_numpy
    
    # you can choose different combinations
    def _strong_aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        if '1' in self.strong_aug_method:
            data_numpy = tools.random_spatial_flip(data_numpy)
        if '2' in self.strong_aug_method:
            data_numpy = tools.random_rotate(data_numpy)
        if '3' in self.strong_aug_method:
            data_numpy = tools.gaus_noise(data_numpy, std=0.05)
        if '4' in self.strong_aug_method:
            data_numpy = tools.gaus_filter(data_numpy)
        if '5' in self.strong_aug_method:
            data_numpy = tools.axis_mask(data_numpy)
        if '6' in self.strong_aug_method:
            data_numpy = tools.random_time_flip(data_numpy)
        if '7' in self.strong_aug_method:
            data_numpy = tools.dropout(data_numpy)
        if '8' in self.strong_aug_method:
            data_numpy = tools.interpolate(data_numpy, 1)
        if '9' in self.strong_aug_method:
            data_numpy = tools.rescale(data_numpy, 0.5)

        return data_numpy
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import sys, pickle


class ShortcutCIFAR100(data.Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

            for index in range(50000):
                if self.train_labels[index] == 0:                   # adding synthetic shortcut to the first class
                    self.train_data[index][3:7:, 3:7:, ::] = 0

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            for index in range(10000):
                if self.test_labels[index] == 0:         # same operation, adding synthetic shortcut to the first class
                    self.test_data[index][3:7:, 3:7:, ::] = 0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class CIFAR99(data.Dataset):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]

    def __init__(self, root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform


        f = self.test_list[0][0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        self.test_data = entry['data']
        if 'labels' in entry:
            self.test_labels = entry['labels']
        else:
            self.test_labels = entry['fine_labels']
        fo.close()
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

        del_list = []
        for index in range(10000):
            if self.test_labels[index] == 0:
                del_list.append(index)                               # delete the first class
            else:
                self.test_data[index][3:7:, 3:7:, ::] = 0            # adding synthetic shortcuts to all the other cls
        self.test_data = np.delete(self.test_data, del_list, axis=0)
        self.test_labels = np.array(self.test_labels)
        self.test_labels = np.delete(self.test_labels, del_list)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.test_data)
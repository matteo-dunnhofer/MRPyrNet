import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from ..modules import utils as ut


class MRNetDataset(data.Dataset):

    def __init__(self, root_dir, task, plane, train=True, transform=False, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        
        if weights is None:
            neg_weight = np.mean(self.labels)
            self.weights = torch.FloatTensor([neg_weight, 1 - neg_weight])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])

        label = torch.FloatTensor([self.labels[index]])

        weight = torch.FloatTensor([self.weights[self.labels[index]]])

        if self.transform:
            array = ut.random_shift(array, 25)
            array = ut.random_rotate(array, 25)
            array = ut.random_flip(array)

        array = (array - 58.09) / 49.73
        array = np.stack((array,)*3, axis=1)
        array = torch.FloatTensor(array)

        return array, label, weight


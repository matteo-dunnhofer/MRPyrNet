import sys
sys.path.append('..')

import os
import random
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import modules.utils as ut


class MRNetDataset(data.Dataset):
    
    def __init__(self, root_dir, task, plane, train=True, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
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

            pos_weight = np.mean(self.labels)
            if pos_weight > 0.5:
                max_weight = pos_weight
                min_weight = 1 - pos_weight
                self.weights = [1.0 + (min_weight / max_weight), 1.0]
            else:
                max_weight = 1 - pos_weight
                min_weight = pos_weight
                self.weights = [1.0, 1.0 + (min_weight / max_weight)]

            self.weights = torch.FloatTensor(self.weights)
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        
        label = torch.LongTensor([self.labels[index]])

        if self.train:
            array = ut.random_shift(array, 25)
            array = ut.random_resize(array, 0.9, 1.1)
            array = ut.random_rotate(array, 10)
            array = ut.random_flip(array)
        
            if self.plane == 'axial' or self.plane == 'coronal':
                if random.random() < 0.5:
                    array = ut.rotate_volume(array, random.choice([0, 1, 2, 3]) * 90)

        array = (array - 58.09) / 49.73

        array = torch.FloatTensor(array) 

        return array, label


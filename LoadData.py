"""
LoadData class for preparing news cascades for model training
"""

# load libraries
import os
import numpy as np
import torch
from torch_geometric.data import DataLoader


# create data loader class object
class LoadData:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.train_size = 500  # NB: change this later on
        self.trainset, self.testset = [], []
        self.files = os.listdir(self.data_folder)
        self.data_objects = self._get_data_objects()

    def _get_data_objects(self):
        i=0
        for file in self.files:
            filename = os.path.join(self.data_folder, file)
            data_obj = torch.load(filename) 

            # convert tensor types; another way??
            data_obj.x = data_obj.x.to(torch.float)
            data_obj.y = data_obj.y.to(torch.long)
            
            # limit train and test set to n=500
            if i < self.train_size:
                self.trainset.append(data_obj)
            elif i == 1000: # remove this later on 
                break
            elif i >= self.train_size:
                self.testset.append(data_obj)
            i += 1
        return self.trainset, self.testset

    def load_train_data(self, batch_size=1):
        return DataLoader(self.trainset, batch_size=batch_size)

    def load_test_data(self, batch_size=1):
        return DataLoader(self.testset, batch_size=batch_size)

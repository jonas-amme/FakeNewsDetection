"""
LoadData class for preparing news cascades for model training
"""

# load libraries
import os
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import DataLoader
from torch.utils.data import random_split


# create data loader class object
class LoadData:
    def __init__(self, data_folder):
        self.data_folder = data_folder
        # self.train_size = 500  # TODO: change this later on
        self.real_data, self.fake_data, self.graph_data = [], [], []
        self.files = os.listdir(self.data_folder)
        self.data_objects = self._get_data_objects()

    def _get_data_objects(self):
        fake = 0
        real = 0
        # TODO: add tqdm and print statements
        print('====== Start collecting cascades ======')
        for file in tqdm(self.files):
            filename = os.path.join(self.data_folder, file)
            data_obj = torch.load(filename)
            if len(data_obj.x) == 0:
                continue
            if data_obj.y == 1: # TODO: remove constraints to load full data
                # if real == 1000:
                #     continue
                self.real_data.append(data_obj)
                real += 1
            else: # TODO: remove constraints to load full data
                # if fake == 1000:
                #     continue
                self.fake_data.append(data_obj)
                fake += 1
            self.graph_data.append(data_obj)
        return self.real_data, self.fake_data, self.graph_data

    # TODO: think of useful functions for data sets
    # def load_graph_data(self, batch_size=1):
    #     return DataLoader(self.graph_data, batch_size=batch_size, shuffle=True)
    #
    # def load_real_data(self, batch_size=1):
    #     return DataLoader(self.real_data, batch_size=batch_size)
    #
    # def load_fake_data(self, batch_size=1):
    #     return DataLoader(self.fake_data, batch_size=batch_size)
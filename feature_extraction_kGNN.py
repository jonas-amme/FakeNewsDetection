import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv

from sklearn.manifold import TSNE
import umap

from LoadData import LoadData
from preprocess import normalizeFeatures

num_features = 14
nhid = 128

# Load Data
data_path = os.getcwd()
graph_data = torch.load(data_path + '/graph_data.pt')

# preprocess graph data
graph_data = normalizeFeatures(graph_data, as_baseline=True)


# kGNN
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.num_features = num_features
        self.nhid = nhid

        self.conv1 = GraphConv(self.num_features, self.nhid * 2)
        self.conv2 = GraphConv(self.nhid * 2, self.nhid * 2)

        # self.fc1 = Linear(self.nhid * 2, self.nhid)
        # self.fc2 = Linear(self.nhid, 2)

    def forward(self, x, edge_index, batch):
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        # x = F.selu(global_mean_pool(x, batch))
        # x = F.selu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.fc2(x)
        return x


model = FeatureExtractor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
param_groups = optimizer.state_dict()['param_groups']

checkpoint = torch.load(os.path.join(data_path, 'model', 'kGNN.pt'), map_location=torch.device('cpu'))
dropped_layers = ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
dropped_params = [6, 7, 8, 9]

feature_state_dict = dict()
for key, val in checkpoint['model_state_dict'].items():
    if key not in dropped_layers:
        feature_state_dict[key] = val

optimizer_state_dict = dict()
optimizer_state_dict['param_groups'] = param_groups
optimizer_state_dict['state'] = dict()
for key in checkpoint['optimizer_state_dict']['state']:
    if key not in dropped_params:
        optimizer_state_dict['state'][key] = checkpoint['optimizer_state_dict']['state'][key]

# load new states
model.load_state_dict(feature_state_dict)
optimizer.load_state_dict(optimizer_state_dict)

# initialize list of features
features_list = list()
labels_list = list()

# create dataloader object
dataloader = DataLoader(graph_data, batch_size=1)

# compute features at last convolutional layer
for data in dataloader:
    with torch.no_grad():
        label = data.y
        try:
            feature = model(data.x, data.edge_index, data.batch)
        except AssertionError:
            continue
        features_list.append(np.array(feature))
        labels_list.extend(np.repeat(label, len(feature)))


# stack features vertically for vertex wise results
features_data = np.vstack(features_list)


# compute tsne embedding
def show_TSNE_embedding(features, labels_list, legend=['Fake', 'True']):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features_tsne = tsne.fit_transform(features)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                          c=labels_list, edgecolor='none', alpha=0.4,
                          cmap=plt.cm.get_cmap('viridis', 2))
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, legend)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('T-SNE embedding of vertex-wise features produced \n at last graph convolutional layer')
    plt.show()


# show results
show_TSNE_embedding(features_data, labels_list)

# save figure
plt.savefig(os.path.join(data_path, 'plots', 'baseline_tsne_kGNN.png'))


# compute umap embedding
def show_UMAP_embedding(features, labels_list, legend=['Fake', 'True']):
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=labels_list, edgecolor='none', alpha=0.4,
                          cmap=plt.cm.get_cmap('viridis', 2))
    plt.gca().set_aspect('equal', 'datalim')
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, legend)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('UMAP embedding of vertex-wise features produced \n at last graph convolutional layer')
    plt.show()


# show results
show_UMAP_embedding(features_data, labels_list)

# save figure
plt.savefig(os.path.join(data_path, 'plots', 'is_umap_kGNN.png'))

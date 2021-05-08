import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

from sklearn.manifold import TSNE

from LoadData import LoadData
from preprocess import normalizeFeatures


# Load Data
data_path = os.getcwd()
dataloader = LoadData('../00_Data/simple_cascades/output')
graph_data = dataloader.graph_data

# preprocess graph data
graph_data = normalizeFeatures(graph_data)

# GIN + TopKPooling
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.num_features = 14
        self.nhid = 128

        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.pool1 = TopKPooling(self.nhid, ratio=0.8)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.pool2 = TopKPooling(self.nhid, ratio=0.8)

        # self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        # self.lin2 = torch.nn.Linear(self.nhid, 32)
        # self.lin3 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = x1 + x2
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.lin2(x))
        # x = F.log_softmax(self.lin3(x), dim=-1)
        return x


# create feature extractor
model = FeatureExtractor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
param_groups = optimizer.state_dict()['param_groups']

# load trained model and remove unused layer
checkpoint = torch.load(os.path.join(data_path, 'model', 'GIN_TopK.pt'), map_location=torch.device('cpu'))
dropped_layers = ["lin1.weight", "lin1.bias", "lin2.weight", "lin2.bias", "lin3.weight", "lin3.bias"]
dropped_params = [8, 9, 10, 11, 12, 13]

# create new model state dictionaries
feature_state_dict = dict()
for key, val in checkpoint['model_state_dict'].items():
    if key not in dropped_layers:
        feature_state_dict[key] = val

# create new optimizer state dictionaries
optimizer_state_dict = dict()
optimizer_state_dict['param_groups'] = param_groups
optimizer_state_dict['state'] = dict()
for key in checkpoint['optimizer_state_dict']['state']:
    if key not in dropped_params:
        optimizer_state_dict['state'][key] = checkpoint['optimizer_state_dict']['state'][key]

# load new state dictionaries
model.load_state_dict(feature_state_dict)
optimizer.load_state_dict(optimizer_state_dict)

# intitialize new feature and label list
features = list()
labels = list()

# create data loader object
dataloader = DataLoader(graph_data, batch_size=1)

# extract features at last pooling layer
for data in dataloader:
    with torch.no_grad():
        label = data.y
        feature = model(data.x, data.edge_index, data.batch)
        features.append(np.array(feature))
        labels.append(label)

# convert to array
features_data = np.array(features).reshape((len(graph_data), features[0].shape[1]))
features_labels = np.array(labels)


# compute tsne embedding
def show_TSNE_embedding(features, labels_list):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features_tsne = tsne.fit_transform(features)
    legend = ['Fake', 'True']
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                          c=labels_list, edgecolor='none', alpha=0.4,
                          cmap=plt.cm.get_cmap('viridis', 2))
    handles, _ = scatter.legend_elements(prop='colors')
    plt.legend(handles, legend)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('T-SNE embedding of cascade-wise features produced \n at last pooling layer')
    plt.show()


# plot results
show_TSNE_embedding(features_data, features_labels)


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

num_features = 14
nhid = 128

# GCN
class Net1(torch.nn.Module):
    def __init__(self, name):
        super(Net1, self).__init__()

        self.name = name
        self.num_features = num_features
        self.nhid = nhid

        self.conv1 = GCNConv(self.num_features, self.nhid * 2)
        self.conv2 = GCNConv(self.nhid * 2, self.nhid * 2)

        self.fc1 = Linear(self.nhid * 2, self.nhid)
        self.fc2 = Linear(self.nhid, 2)

    def forward(self, x, edge_index, batch):
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# GIN
class Net2(torch.nn.Module):
    def __init__(self, name):
        super(Net2, self).__init__()

        self.name = name
        self.num_features = num_features
        self.nhid = nhid

        self.conv1 = GraphConv(self.num_features, self.nhid * 2)
        self.conv2 = GraphConv(self.nhid * 2, self.nhid * 2)

        self.fc1 = Linear(self.nhid * 2, self.nhid)
        self.fc2 = Linear(self.nhid, 2)

    def forward(self, x, edge_index, batch):
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# GIN + TopKPooling
class Net3(torch.nn.Module):
    def __init__(self, name):
        super(Net3, self).__init__()

        self.name = name
        self.num_features = num_features
        self.nhid = nhid

        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.pool1 = TopKPooling(self.nhid, ratio=0.8)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.pool2 = TopKPooling(self.nhid, ratio=0.8)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, 32)
        self.lin3 = torch.nn.Linear(32, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = x1 + x2
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x


class Net4(torch.nn.Module):
    def __init__(self, name):
        super(Net4, self).__init__()

        self.name = name
        self.num_features = num_features
        self.nhid = nhid

        self.conv1 = GATConv(self.num_features, self.nhid * 2)
        self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

        self.fc1 = Linear(self.nhid * 2, self.nhid)
        self.fc2 = Linear(self.nhid, 2)

    def forward(self, x, edge_index, batch):
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


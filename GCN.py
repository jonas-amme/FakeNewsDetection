"""
Reimplementation of GCNN model from paper Fake News Detection on Social Media using Geometric Deep Learning.
Forked from https://github.com/YingtongDou/GCNN
Original paper https://arxiv.org/abs/1902.06673
"""

# load libraries
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

from LoadData import LoadData


# Create the model 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.num_features = 9
        self.nhid = 64

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


def eval(log):

    accuracy, f1_macro, precision, recall = 0, 0, 0, 0

    prob_log, label_log = [], []

    for batch in log:
        pred_y, y = batch[0].data.cpu().numpy().argmax(axis=1), batch[1].data.cpu().numpy().tolist()
        prob_log.extend(batch[0].data.cpu().numpy()[:, 1].tolist())
        label_log.extend(y)

        accuracy += accuracy_score(y, pred_y)
        f1_macro += f1_score(y, pred_y, average='macro')
        precision += precision_score(y, pred_y, zero_division=0)
        recall += recall_score(y, pred_y, zero_division=0)

    return accuracy / len(log), f1_macro / len(log), precision / len(log), recall / len(log)


def test(loader):
    model.eval()

    loss_test = 0.0
    out_log = []
    with torch.no_grad():
        for data in loader:
            data = data
            out = model(data.x, data.edge_index, data.batch)
            y = data.y
            out_log.append([F.softmax(out, dim=1), y])
            loss_test += F.nll_loss(out, y).item()
    return eval(out_log), loss_test


# Load Data
DATA_PATH = "../00_Data/cascades/cascades"
dataloader = LoadData(DATA_PATH)
trainset = dataloader.load_train_data()
testset = dataloader.load_test_data()

# view some data
for step, data in enumerate(trainset):
    if step == 4:
        break
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

# initialize model
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# model training
epochs = 10
out_log = []

model.train()
for epoch in tqdm(range(epochs)):
    loss_train = 0.0
    correct = 0
    for i, data in enumerate(trainset):
        optimizer.zero_grad()
        data = data
        out = model(data.x, data.edge_index, data.batch)
        y = data.y
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        out_log.append([F.softmax(out, dim=1), y])
    acc_train, _, _, recall_train = eval(out_log)
    [acc_val, _, _, recall_val], loss_val = test(testset)
    print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
          f' recall_train: {recall_train:.4f}, loss_val: {loss_val:.4f},'
          f' acc_val: {acc_val:.4f}, recall_val: {recall_val:.4f}')

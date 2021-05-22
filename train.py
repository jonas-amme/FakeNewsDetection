
import time
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score

from LoadData import LoadData
from preprocess import normalizeFeatures


# initialize arguments
parser = argparse.ArgumentParser()

# specify model parameters
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--data_path', type=str, default='/data/s2583550/FakeNewsDetection/simple_cascades/output', help='enter your data path')
parser.add_argument('--model_path', type=str, default='/data/s2583550/FakeNewsDetection/model/', help='enter your model path')
parser.add_argument('--save_name', type=str, default='default', help='enter save name for model state dict')
parser.add_argument('--as_baseline', type=bool, default=False, help='use IS cascades or baseline cascades')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--num_features', type=int, default=15, help='number of features (14 for baseline)')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# view arguments
print(args)


# Load Data
dataloader = LoadData(args.data_path)  # args.data_path = data path of twitter data, to be specified in beginning
graph_data = dataloader.graph_data

# preprocess graph data
graph_data = normalizeFeatures(graph_data, as_baseline=args.as_baseline)

# create train, val test split
num_training = int(len(graph_data) * 0.7)
num_val = int(len(graph_data) * 0.1)
num_test = len(graph_data) - (num_training + num_val)
training_set, validation_set, test_set = random_split(graph_data, [num_training, num_val, num_test],
                                                      generator=torch.Generator().manual_seed(42))

# create dataloader objects
train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)


# create models

# GCN
class Net1(torch.nn.Module):
    def __init__(self, name):
        super(Net1, self).__init__()

        self.name = name
        self.num_features = args.num_features
        self.nhid = args.nhid

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)

        self.fc1 = Linear(self.nhid, self.nhid)
        self.fc2 = Linear(self.nhid, 2)

    def forward(self, x, edge_index, batch):
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# kGNN
class Net2(torch.nn.Module):
    def __init__(self, name):
        super(Net2, self).__init__()

        self.name = name
        self.num_features = args.num_features
        self.nhid = args.nhid

        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.conv2 = GraphConv(self.nhid, self.nhid)

        self.fc1 = Linear(self.nhid, self.nhid)
        self.fc2 = Linear(self.nhid, 2)

    def forward(self, x, edge_index, batch):
        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

# kGNN + TopKPooling
class Net3(torch.nn.Module):
    def __init__(self, name):
        super(Net3, self).__init__()

        self.name = name
        self.num_features = args.num_features
        self.nhid = args.nhid

        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.pool1 = TopKPooling(self.nhid, ratio=0.8)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.pool2 = TopKPooling(self.nhid, ratio=0.8)

        self.lin1 = torch.nn.Linear(self.nhid*2, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, 16)
        self.lin3 = torch.nn.Linear(16, 2)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = F.relu(self.conv2(x, edge_index))nhi
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        x = x1 + x2
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x

# GAT
class Net4(torch.nn.Module):
    def __init__(self, name):
        super(Net4, self).__init__()

        self.name = name
        self.num_features = args.num_features
        self.nhid = args.nhid

        self.conv1 = GATConv(self.num_features, self.nhid)
        self.conv2 = GATConv(self.nhid, self.nhid)

        self.fc1 = Linear(self.nhid, self.nhid)
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

    return label_log, prob_log, accuracy / len(log), f1_macro / len(log), precision / len(log), recall / len(log)


def compute_test(loader):
    model.eval()
    loss_test = 0.0
    out_log = []
    with torch.no_grad():
        for data in loader:
            data = data.to(args.device)
            out = model(data.x, data.edge_index, data.batch)
            y = data.y
            out_log.append([F.softmax(out, dim=1), y])
            loss_test += F.nll_loss(out, y).item()
    return eval(out_log), loss_test


# create models
m1 = Net1(name='GCN')
m2 = Net2(name='kGNN')
m3 = Net3(name='kGNN_TopK')
m4 = Net4(name='GAT')
models = [m1, m2, m3, m4]


print('Start training ... \n')

if __name__ == '__main__':

    # model training
    for model in models:
        print(f'============= {model.name} =============')

        # to GPU
        model.to(args.device)

        # initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        out_log = list()
        loss_train_list = list()
        loss_val_list = list()
        acc_train_list = list()
        acc_val_list = list()

        t = time.time()
        model.train()
        for epoch in tqdm(range(args.epochs)):
            loss_train = 0.0
            correct = 0
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                data = data.to(args.device)
                out = model(data.x, data.edge_index, data.batch)
                y = data.y
                loss = F.nll_loss(out, y)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
                out_log.append([F.softmax(out, dim=1), y])

            # model validation
            _, _, acc_train, _, _, recall_train = eval(out_log)
            [_, _, acc_val, f1_macro, precision, recall_val], loss_val = compute_test(val_loader)

            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
            acc_train_list.append(acc_train)
            acc_val_list.append(acc_val)

            print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
                  f' recall_train: {recall_train:.4f}, loss_val: {loss_val:.4f},'
                  f' acc_val: {acc_val:.4f}, recall_val: {recall_val:.4f}')

        # create model dictionary
        model_dict = {
            'name': model.name,
            'epochs': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train_list,
            'loss_val': loss_val_list,
            'acc_train': acc_train_list,
            'acc_val': acc_val_list
        }

        # save trained model
        torch.save(model_dict, args.model_path + args.save_name + model.name + '.pt')

        # model test
        [_, _, acc, f1_macro, precision, recall], test_loss = compute_test(test_loader)
        print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, '
              f'precision: {precision:.4f}, recall: {recall:.4f}')
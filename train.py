
import time
from tqdm import tqdm
import argparse
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score

from LoadData import LoadData
from preprocess import normalizeFeatures
from models import *


# initialize arguments
parser = argparse.ArgumentParser()

# specify model parameters
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--data_path', type=str, default='/data/s2583550/FakeNewsDetection/simple_cascades/output', help='enter your data path')
parser.add_argument('--model_path', type=str, default='/data/s2583550/FakeNewsDetection/model/', help='enter your model path')
parser.add_argument('--save_name', type=str, default='default', help='enter save name for model state dict')
parser.add_argument('--as_baseline', type=bool, default=False, help='use IS cascades or baseline cascades')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=32, help='hidden size')
parser.add_argument('--num_features', type=int, default=15, help='number of features (14 for baseline)')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# view arguments
print(args)



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


# load data
dataloader = LoadData(args.data_path)  # args.data_path = data path of twitter data, to be specified in beginning
graph_data = dataloader.graph_data

# preprocess graph data
graph_data = normalizeFeatures(graph_data, as_baseline=args.as_baseline)

# randomly shuffle graph data
random.seed(42)
random.shuffle(graph_data)

# create train/test split
ntrain = int(len(graph_data) * 0.8)
train_set = graph_data[:ntrain]
test_set = graph_data[ntrain:]

# initialize cross-validation
K = 5
folds = list()
fold_size = round(ntrain / K)
i = 0
start = 0
end = fold_size
while i < K:
    folds.append(train_set[start:end])
    start += fold_size
    end += fold_size
    i += 1

# create test dataloader object
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


print('Start training ... \n')

if __name__ == '__main__':

    # start cross-validation
    for k in range(K):

        # create train and validation folds
        train_folds = list()
        val_fold = list()

        for j in range(K):
            if j == k:
                val_fold = folds[j]
            else:
                train_folds.extend(folds[j])

        # create dataloader objects
        train_loader = DataLoader(train_folds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=args.batch_size, shuffle=True)

        # create models
        m1 = Net1(name='GCN', nhid=args.nhid, num_features=args.num_features)
        m2 = Net2(name='kGNN', nhid=args.nhid, num_features=args.num_features)
        m3 = Net3(name='kGNN_TopK', nhid=args.nhid, num_features=args.num_features)
        models = [m1, m2, m3]

    # model training
    for model in models:
        print(f'============= model: {model.name}, fold: {str(k)}=============')

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
            'fold': k,
            'epochs': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_train': loss_train_list,
            'loss_val': loss_val_list,
            'acc_train': acc_train_list,
            'acc_val': acc_val_list
        }

        # save trained model
        torch.save(model_dict, args.model_path + args.save_name + model.name + '_fold' + str(k) + '.pt')

        # model test
        [_, _, acc, f1_macro, precision, recall], test_loss = compute_test(test_loader)
        print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, '
              f'precision: {precision:.4f}, recall: {recall:.4f}')
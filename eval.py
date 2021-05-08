import os

import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from LoadData import LoadData
from preprocess import normalizeFeatures
from models import Net1, Net2, Net3, Net4

# Load Data
dataloader = LoadData('../00_Data/simple_cascades/output')
graph_data = dataloader.graph_data
data_path = os.getcwd()

# preprocess graph data
graph_data = normalizeFeatures(graph_data)

# set seed
torch.manual_seed(777)  # reproduces the same data split as during training

# create train, val test split
num_training = int(len(graph_data) * 0.6)
num_val = int(len(graph_data) * 0.2)
num_test = len(graph_data) - (num_training + num_val)
training_set, validation_set, test_set = random_split(graph_data, [num_training, num_val, num_test])

# create dataloader objects
train_loader = DataLoader(training_set, batch_size=32, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# create models
m1 = Net1(name='GCN')
m2 = Net2(name='GIN')
m3 = Net3(name='GIN_TopK')
m4 = Net4(name='GAT')
models = [m1, m2, m3, m4]


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
            # data = data.to(args.device)
            out = model(data.x, data.edge_index, data.batch)
            y = data.y
            out_log.append([F.softmax(out, dim=1), y])
            loss_test += F.nll_loss(out, y).item()
    return eval(out_log), loss_test


# initialize list of scores
scores_list = list()

# evaluate all models
for model in models:
    scores = dict()

    # load the model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    checkpoint = torch.load(os.path.join(data_path, 'model', model.name + '.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # add checkpoint to scores dict
    scores['name'] = model.name
    scores['checkpoint'] = checkpoint

    # evaluate on test set
    [label_log, prob_log, acc, f1_macro, precision, recall], test_loss = compute_test(test_loader)
    print(f'============ {model.name} ============')
    print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, '
          f'precision: {precision:.4f}, recall: {recall:.4f}')

    # add test scores to scores
    scores['label_log'] = label_log
    scores['test_prob_log'] = prob_log
    scores['test_acc'] = acc
    scores['test_f1'] = f1_macro
    scores['test_precision'] = precision
    scores['test_recall'] = recall
    scores['test_loss'] = test_loss

    # scores to list
    scores_list.append(scores)


# add baseline classifier
def extract_avg_feature(data):
    dataset_avg_features = []
    for cascade_index, cascade in enumerate(data):
        feature_values = np.zeros(shape=(16))
        for node_index, node_feature_tensor in enumerate(cascade.x):
            for feature_index, feature_value in enumerate(node_feature_tensor):
                feature_values[feature_index] += feature_value

        try:
            feature_values = feature_values / len(cascade.x)
        except:
            continue
        d = {
            'location': feature_values[0],
            'profile_description': feature_values[1],
            'account_creation': feature_values[2],
            'verified': feature_values[3],
            'favorites_count': feature_values[4],
            'listed_count': feature_values[5],
            'status_count': feature_values[6],
            'followers': feature_values[7],
            'friends': feature_values[8],
            'retweet_time': feature_values[9],
            'source_device': feature_values[10],
            'retweets': feature_values[11],
            'favorites': feature_values[12],
            'text': feature_values[13],
            'hashtags': feature_values[14],
            'interaction_strength': feature_values[15],
            'label': cascade.y.item()
        }
        dataset_avg_features.append(d)
    dataset_avg_features = pd.DataFrame(dataset_avg_features)
    return dataset_avg_features


# extract dataset of average features
avg_graph_data = extract_avg_feature(graph_data)

# create train test split
train, test = train_test_split(avg_graph_data, test_size=0.2)
x_train, x_test = train.drop('label', axis=1), test.drop('label', axis=1)
y_train, y_test = train['label'], test['label']

# perform logistic regression
score_logreg = dict()
score_logreg['name'] = 'Logistic Regression'
score_logreg['label_log'] = y_test
clf_logreg = LogisticRegression(random_state=0).fit(x_train, y_train)
score_logreg['test_prob_log'] = clf_logreg.predict_proba(x_test)[:, 1]
scores_list.append(score_logreg)

# perform random forest
score_rf = dict()
score_rf['name'] = 'Random Forest'
score_rf['label_log'] = y_test
clf_rf = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
score_rf['test_prob_log'] = clf_rf.predict_proba(x_test)[:, 1]
scores_list.append(score_rf)


# plot results

# plot performance during training
def show_training(model_list):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    for model_list in scores_list:
        try:
            axes[0, 0].plot(model_list['checkpoint']['loss_train'], label=model_list['checkpoint']['name'])
            axes[0, 0].set_title('Training loss')

            axes[0, 1].plot(model_list['checkpoint']['acc_train'], label=model_list['checkpoint']['name'])
            axes[0, 1].set_title('Training accuracy')
            axes[0, 1].legend(loc='lower right', prop={'size': 8})

            axes[1, 0].plot(model_list['checkpoint']['loss_val'], label=model_list['checkpoint']['name'])
            axes[1, 0].set_title('Validation loss')

            axes[1, 1].plot(model_list['checkpoint']['acc_val'], label=model_list['checkpoint']['name'])
            axes[1, 1].set_title('Validation accuracy')
        except KeyError:
            continue
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()


show_training(scores_list)


# compute AUC
def show_roc_curves(model_list):
    plt.figure()
    for model in model_list:
        auc = roc_auc_score(model['label_log'], model['test_prob_log'])
        fpr, tpr, tr = roc_curve(model['label_log'], model['test_prob_log'])
        plt.plot(fpr, tpr, label=f"{model['name']} (AUC {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Line of no discrimination", color='grey')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'$1-S_p$')
    plt.ylabel(r'$S_e$')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.show()


show_roc_curves(scores_list)


def show_calibration_curve(model_list):
    plt.figure()
    for model in model_list:
        y, x = calibration_curve(model['label_log'], model['test_prob_log'])
        plt.plot(x, y, marker='o', label=model['name'])
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Perfect calibration')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    plt.legend(loc="upper left", prop={'size': 8})
    sns.rugplot(prob_log, axis='x', color='red')
    plt.show()


show_calibration_curve(scores_list)

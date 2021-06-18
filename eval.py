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
import itertools

from LoadData import LoadData
from preprocess import normalizeFeatures
from models import Net1, Net2, Net3, Net4

# Load Data
data_path = os.getcwd()
graph_data = torch.load(data_path + '/graph_data.pt')

# preprocess graph data
baseline_data = normalizeFeatures(graph_data, as_baseline=True)
is_data = normalizeFeatures(graph_data, as_baseline=False)

# set seed
torch.manual_seed(777)

# create train, val test split
num_training = int(len(graph_data) * 0.7)
num_val = int(len(graph_data) * 0.1)
num_test = len(graph_data) - (num_training + num_val)

# split baseline data
_, _, base_test_set = random_split(baseline_data, [num_training, num_val, num_test],
                                   generator=torch.Generator().manual_seed(42))

# split is data
_, _, is_test_set = random_split(is_data, [num_training, num_val, num_test],
                                 generator=torch.Generator().manual_seed(42))

# create dataloader objects
base_test_loader = DataLoader(base_test_set, batch_size=32, shuffle=False)
is_test_loader = DataLoader(is_test_set, batch_size=32, shuffle=False)

# create models
m1 = Net1(name='GCN')
m2 = Net2(name='kGNN')
m3 = Net3(name='kGNN_TopK')
m4 = Net4(name='GAT')
baseline_models = [m1, m2, m3, m4]

m5 = Net1(name='IS_GCN')
m6 = Net2(name='IS_kGNN')
m7 = Net3(name='IS_kGNN_TopK')
m8 = Net4(name='IS_GAT')
is_models = [m5, m6, m7, m8]


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


def compute_test(model, loader):
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


def evaluate_models(model_list, test_loader):
    # initialize list of scores
    scores_list = list()

    for model in model_list:
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
        [label_log, prob_log, acc, f1_macro, precision, recall], test_loss = compute_test(model, test_loader)
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
        scores['test_auc'] = roc_auc_score(label_log, prob_log)

        # scores to list
        scores_list.append(scores)

    return scores_list


# evaluate models
baseline_scores = evaluate_models(baseline_models, base_test_loader)
is_scores = evaluate_models(is_models, is_test_loader)


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
train, test = train_test_split(avg_graph_data, test_size=0.2, random_state=0)
x_train, x_test = train.drop('label', axis=1), test.drop('label', axis=1)
y_train, y_test = train['label'], test['label']

# perform logistic regression
score_logreg = dict()
score_logreg['name'] = 'Logistic Regression'
score_logreg['label_log'] = y_test
clf_logreg = LogisticRegression(random_state=0).fit(x_train, y_train)
score_logreg['test_prob_log'] = clf_logreg.predict_proba(x_test)[:, 1]
y_test_pred = clf_logreg.predict(x_test)
score_logreg['test_acc'] = accuracy_score(y_test, y_test_pred)
score_logreg['test_f1'] = f1_score(y_test, y_test_pred, average='macro')
score_logreg['test_precision'] = precision_score(y_test, y_test_pred, zero_division=0)
score_logreg['test_recall'] = recall_score(y_test, y_test_pred, zero_division=0)
score_logreg['test_auc'] = roc_auc_score(y_test, score_logreg['test_prob_log'])
baseline_scores.append(score_logreg)
is_scores.append(score_logreg)


# perform random forest
score_rf = dict()
score_rf['name'] = 'Random Forest'
score_rf['label_log'] = y_test
clf_rf = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
score_rf['test_prob_log'] = clf_rf.predict_proba(x_test)[:, 1]
y_test_pred = clf_rf.predict(x_test)
score_rf['test_acc'] = accuracy_score(y_test, y_test_pred)
score_rf['test_f1'] = f1_score(y_test, y_test_pred, average='macro')
score_rf['test_precision'] = precision_score(y_test, y_test_pred, zero_division=0)
score_rf['test_recall'] = recall_score(y_test, y_test_pred, zero_division=0)
score_rf['test_auc'] = roc_auc_score(y_test, score_rf['test_prob_log'])
baseline_scores.append(score_rf)
is_scores.append(score_rf)


# plot results

# plot performance during training
def show_training(model_list):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    for model in model_list:
        try:
            axes[0, 0].plot(model['checkpoint']['loss_train'], label=model['checkpoint']['name'])
            axes[0, 0].set_title('Training loss')

            axes[0, 1].plot(model['checkpoint']['acc_train'], label=model['checkpoint']['name'])
            axes[0, 1].set_title('Training accuracy')
            axes[0, 1].legend(loc='lower right', prop={'size': 8})

            axes[1, 0].plot(model['checkpoint']['loss_val'], label=model['checkpoint']['name'])
            axes[1, 0].set_title('Validation loss')

            axes[1, 1].plot(model['checkpoint']['acc_val'], label=model['checkpoint']['name'])
            axes[1, 1].set_title('Validation accuracy')
        except KeyError:
            continue
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()


# show results
show_training(baseline_scores)
plt.savefig(os.path.join(data_path, 'plots', 'baseline_training.png'))

# show is results
show_training(is_scores)
plt.savefig(os.path.join(data_path, 'plots', 'is_training.png'))


# plot performance during training
def compare_training(base_list, is_list):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 7))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    for i, model in enumerate(base_list):
        try:
            axes[0, 0].plot(model['checkpoint']['loss_train'], label=model['checkpoint']['name'], color=colors[i])
            axes[0, 0].set_title('Training loss')

            axes[0, 1].plot(model['checkpoint']['acc_train'], label=model['checkpoint']['name'], color=colors[i])
            axes[0, 1].set_title('Training accuracy')
            axes[0, 1].legend(loc='lower right', prop={'size': 8})

            axes[1, 0].plot(model['checkpoint']['loss_val'], label=model['checkpoint']['name'], color=colors[i])
            axes[1, 0].set_title('Validation loss')

            axes[1, 1].plot(model['checkpoint']['acc_val'], label=model['checkpoint']['name'], color=colors[i])
            axes[1, 1].set_title('Validation accuracy')
        except KeyError:
            continue
    for i, model in enumerate(is_list):
        try:
            axes[0, 0].plot(model['checkpoint']['loss_train'], '--', label=model['checkpoint']['name'], color=colors[i])
            axes[0, 0].set_title('Training loss')

            axes[0, 1].plot(model['checkpoint']['acc_train'], '--', label=model['checkpoint']['name'], color=colors[i])
            axes[0, 1].set_title('Training accuracy')
            axes[0, 1].legend(loc='lower right', prop={'size': 7})

            axes[1, 0].plot(model['checkpoint']['loss_val'], '--', label=model['checkpoint']['name'], color=colors[i])
            axes[1, 0].set_title('Validation loss')

            axes[1, 1].plot(model['checkpoint']['acc_val'], '--', label=model['checkpoint']['name'], color=colors[i])
            axes[1, 1].set_title('Validation accuracy')
        except KeyError:
            continue
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.show()


# show results
compare_training(baseline_scores, is_scores)
plt.savefig(os.path.join(data_path, 'plots', 'compare_training.png'))


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


# show results
show_roc_curves(baseline_scores)
plt.savefig(os.path.join(data_path, 'plots', 'baseline_auc.png'))

# show results
show_roc_curves(is_scores)
plt.savefig(os.path.join(data_path, 'plots', 'is_auc.png'))


# compute AUC
def compare_roc_curves(base_list, is_list):
    plt.figure(figsize=(8, 6))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:6]
    for i, model in enumerate(base_list[:4]):
        auc = roc_auc_score(model['label_log'], model['test_prob_log'])
        fpr, tpr, tr = roc_curve(model['label_log'], model['test_prob_log'])
        plt.plot(fpr, tpr, label=f"{model['name']} (AUC {auc:.2f})", color=colors[i])
    for i, model in enumerate(is_list[:4]):
        auc = roc_auc_score(model['label_log'], model['test_prob_log'])
        fpr, tpr, tr = roc_curve(model['label_log'], model['test_prob_log'])
        plt.plot(fpr, tpr, '--', label=f"{model['name']} (AUC {auc:.2f})", color=colors[i])
    for i, model in enumerate(base_list[4:]):
        auc = roc_auc_score(model['label_log'], model['test_prob_log'])
        fpr, tpr, tr = roc_curve(model['label_log'], model['test_prob_log'])
        plt.plot(fpr, tpr, label=f"{model['name']} (AUC {auc:.2f})", color=colors[i + 4])
    plt.plot([0, 1], [0, 1], linestyle='--', label="Line of no discrimination", color='grey')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel(r'$1-S_p$')
    plt.ylabel(r'$S_e$')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.show()


# show results
compare_roc_curves(baseline_scores, is_scores)
plt.savefig(os.path.join(data_path, 'plots', 'compare_auc.png'))


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
    plt.legend(loc="lower right", bbox_to_anchor=(0.95, 0.05), prop={'size': 8})
    plt.show()


# show results
show_calibration_curve(baseline_scores)
plt.savefig(os.path.join(data_path, 'plots', 'baseline_calibration.png'))

# show results
show_calibration_curve(is_scores)
plt.savefig(os.path.join(data_path, 'plots', 'is_calibration.png'))



# create table of results
baseline_scores[0].keys()

keys = ['name', 'test_acc', 'test_f1', 'test_precision', 'test_recall', 'test_auc']
names = ['Method', 'ACC', 'F1', 'Precision', 'Recall', 'AUC']

all_scores = dict()
for key in keys:
    all_scores[key] = list()
    for model in baseline_scores[:4]:
        all_scores[key].append(model[key])
    for model in is_scores:
        all_scores[key].append(model[key])


all_scores_df = pd.DataFrame.from_dict(all_scores).round(decimals=4)
all_scores_df.columns = names
print(all_scores_df.to_latex(index=False))


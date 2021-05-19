import os
import pandas as pd
import numpy as np
import torch
from LoadData import LoadData

# Load Data
data_path = os.getcwd()
graph_data = torch.load(data_path + '/graph_data.pt')

def extract_nodewise_features(data):
    features_list = list()
    for cascade_idx, cascade in enumerate(data):
        for node_idx, node in enumerate(cascade.x):
            feature_vector = np.zeros(shape=(17))
            for feature_idx, feature in enumerate(node):
                feature_vector[feature_idx] += feature
            feature_vector[-1] = cascade.y
            features_list.append(feature_vector)
    return features_list


# extract node-wise features
nodewise_features = extract_nodewise_features(graph_data)
print(len(nodewise_features))
print(nodewise_features[0])

# create labels for columns
labels = ['location', 'profile_description', 'account_creation', 'verified', 'favorites_count', 'listed_count',
          'status_count',
          'followers_count', 'friends_count', 'retweet_time', 'source_device', 'retweets_count', 'favorites_count',
          'text', 'hashtags', 'interaction_strength', 'label']

# create pandas dataframe from node-wise features
df = pd.DataFrame(nodewise_features, columns=labels)

# crate latex table
total_df_summaries = df.drop(['account_creation', 'retweet_time'], axis=1).describe().drop(
    ['count']).transpose().to_latex(float_format='%.2f')
print(total_df_summaries)

# split data according to true/false labels
df_true = df[df['label'] == 1]
df_false = df[df['label'] == 0]

# compute mean and std
df_mean = df.groupby(['label']).mean().transpose()
df_std = df.groupby(['label']).std().transpose()

# combine data
df_comp = pd.concat([df_mean, df_std], axis=1)
print(df_comp.head())


# function to extract cascade-wise features
def extract_avg_feature(data):
    dataset_avg_features = []
    for cascade_index, cascade in enumerate(data):
        feature_values = np.zeros(shape=(16))
        for node_index, node_feature_tensor in enumerate(cascade.x):
            for feature_index, feature_value in enumerate(node_feature_tensor):
                feature_values[feature_index] += feature_value

        with np.errstate(divide='ignore', invalid='ignore'):
            feature_values = feature_values / len(cascade.x)

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


# extract cascade-wise features
cascadewise_features = extract_avg_feature(graph_data)
cascadewise_features.columns = labels

# split data according to true/false labels
cascades_true = cascadewise_features[cascadewise_features['label'] == 1]
cascades_fake = cascadewise_features[cascadewise_features['label'] == 0]

# compute mean and std
cascades_true_mean = cascadewise_features.groupby(['label']).mean().transpose()
cascades_true_std = cascadewise_features.groupby(['label']).std().transpose()

# concatenate mean and std data
cascades_comp = pd.concat([cascades_true_mean, cascades_true_std], axis=1)
print(cascades_comp.head())

# perform t-test
from scipy import stats

# compare cascade-wise features of true and fake news
t, p = stats.ttest_ind(cascades_true.drop(['label'], axis=1), cascades_fake.drop(['label'], axis=1), nan_policy='omit')
cascades_ttest = pd.DataFrame(pd.concat([pd.Series(t), pd.Series(p), pd.Series(p < 0.05)], axis=1))
cascades_comp = pd.concat([cascades_comp.reset_index(), cascades_ttest.reset_index(drop=True)], axis=1)
cascades_comp = cascades_comp.set_index('index')
cascades_comp.columns = ['mean_fake', 'mean_real', 'std_fake', 'std_real', 't', 'p', 'p < 0.05']
print(cascades_comp.drop(['account_creation', 'retweet_time']).to_latex(float_format='%.4f'))


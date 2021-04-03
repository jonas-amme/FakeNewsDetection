import numpy as np
import datetime

import torch
from torch_geometric.data import Data, DataLoader
from LoadData import LoadData


# list of features
# 0 tweet created at           # currently removed but will need for time between tweets as edge features
# 1 tweet txt embedding
# 2 tweet hashtags embedding
# 3 user name embedding
# 4 user description embedding
# 5 user followers count
# 6 user following count
# 7 user listed count
# 8 user created at            # will be removed
# 9 AgeUntilTweet in seconds   # will be added
# 10 time between tweets       # will be added still need to code !!!!


def addAgeUntilTweet(cascade):

    x = list()
    edge_index = cascade.edge_index
    y = cascade.y

    datetimeFormat = '%Y%m%d%H%M%S'

    for i, row in enumerate(cascade.x):
        features = row.numpy()  # extract features of each node
        date_tweet = str(features[0])  # extract tweet creation date
        date_created = str(features[-1])  # extract user creation date
        diff = datetime.datetime.strptime(date_tweet, datetimeFormat) \
               - datetime.datetime.strptime(date_created, datetimeFormat)  # compute time difference
        diff_sec = np.array(diff.days, ndmin=1)
        features = np.delete(features, [0,1,2,3,4,-1])  # remove tweet and user creation date
        x.append(np.concatenate((features, diff_sec)))  # concat and append to new feature list

    x = torch.tensor(x, dtype=torch.float)  # convert new feature list to torch tensor
    cascade = Data(x=x, edge_index=edge_index, y=y)  # create new cascade representation
    return cascade


def normalize(value, mu, sigma):
    if sigma == 0.0:
        return 0.0
    return (value - mu) / sigma


def normalizeCascade(cascade, MeansSigmas):

    x = list()
    edge_index = cascade.edge_index
    y = cascade.y.to(torch.long)

    for node in cascade.x:  # loop over each node in the cascade

        # initialize new feature vector
        normalized = list()

        # normalize each feature with its corresponding mean and sigma
        # normalized.append(normalize(node[0], MeansSigmas['TextEmb'][0], MeansSigmas['TextEmb'][1]))
        # normalized.append(normalize(node[1], MeansSigmas['HashEmb'][0], MeansSigmas['HashEmb'][1]))
        # normalized.append(normalize(node[2], MeansSigmas['NameEmb'][0], MeansSigmas['NameEmb'][1]))
        # normalized.append(normalize(node[3], MeansSigmas['DescEmb'][0], MeansSigmas['DescEmb'][1]))
        normalized.append(normalize(node[0], MeansSigmas['Followers'][0], MeansSigmas['Followers'][1]))
        normalized.append(normalize(node[1], MeansSigmas['Following'][0], MeansSigmas['Following'][1]))
        normalized.append(normalize(node[2], MeansSigmas['Listed'][0], MeansSigmas['Listed'][1]))
        normalized.append(normalize(node[3], MeansSigmas['Age'][0], MeansSigmas['Age'][1]))

        # add normalized vector to new feature representation
        x.append(normalized)

    x = torch.tensor(x, dtype=torch.float)  # convert new feature list to torch tensor
    cascade = Data(x=x, edge_index=edge_index, y=y)  # create new cascade representation
    return cascade


def normalizeNodeFeatures(data):

    # initialize total feature list
    # TextEmb = list()
    # HashEmb = list()
    # NameEmb = list()
    # DescEmb = list()
    Followers = list()
    Following = list()
    Listed = list()
    Age = list()

    # initialize new graph dataset
    graph_dataset = list()
    for cascade in data:  # loop over each cascade in data
        graph_dataset.append(addAgeUntilTweet(cascade))  # create age until tweet and add to new graph data

    # loop over new graph data
    for cascade in graph_dataset:
        # TextEmb.extend([features[0] for features in cascade.x])  # extract text embedding
        # HashEmb.extend([features[1] for features in cascade.x])  # extract hashtag embedding
        # NameEmb.extend([features[2] for features in cascade.x])  # extract name embedding
        # DescEmb.extend([features[3] for features in cascade.x])  # extract description embedding
        Followers.extend([features[0] for features in cascade.x])  # extract follower count
        Following.extend([features[1] for features in cascade.x])  # extract following count
        Listed.extend([features[2] for features in cascade.x])  # extract listed count
        Age.extend([features[3] for features in cascade.x])  # extract age until tweet

    # calculate means and standard deviations
    MeansSigmas = dict()
    # MeansSigmas['TextEmb'] = (np.mean(TextEmb), np.std(TextEmb))
    # MeansSigmas['HashEmb'] = (np.mean(HashEmb), np.std(HashEmb))
    # MeansSigmas['NameEmb'] = (np.mean(NameEmb), np.std(NameEmb))
    # MeansSigmas['DescEmb'] = (np.mean(DescEmb), np.std(DescEmb))
    MeansSigmas['Followers'] = (np.mean(Followers), np.std(Followers))
    MeansSigmas['Following'] = (np.mean(Following), np.std(Following))
    MeansSigmas['Listed'] = (np.mean(Listed), np.std(Listed))
    MeansSigmas['Age'] = (np.mean(Listed), np.std(Listed))

    # normalize each cascade and add to new dataset
    final_graph_dataset = list()
    for cascade in graph_dataset:
        final_graph_dataset.append(normalizeCascade(cascade, MeansSigmas))

    return final_graph_dataset

"""
Functions for preprocessing the twitter data. Some functions are partly taken from
https://github.com/MarionMeyers/fake_news_detection_propagation/blob/master/GDLtopK-GraphConv.py
"""

import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
from torch_geometric.data import Data


# list of features
# 0 tweet created at                # will be removed
# 1 tweet txt embedding
# 2 tweet hashtags embedding
# 3 user name embedding
# 4 user description embedding
# 5 user followers count
# 6 user following count
# 7 user listed count
# 8 user created at                # will be removed
# 9 AgeUntilTweet in days          # will be added
# 10 TimeBetweenTweets in seconds  # currently ignored TODO: code is implemented but training does not work


def understandDate(date):
    datetimeFormat = '%Y%m%d%H%M%S'  # datetime format
    time = str(date.numpy())  # convert tensor to string
    datetime_object = datetime.strptime(time, datetimeFormat)  # create date object
    return datetime_object


def computeTimeDifference(date1, date2):
    date1 = understandDate(date1)
    date2 = understandDate(date2)
    difference = date1 - date2
    return difference


def addTimeBetweenTweets(cascade):
    # ADD TIME BETWEEN TWEETS AS EDGE ATTRIBUTES

    # extract cascade features
    all_time_created = [features[0] for features in cascade.x]
    time_origin = all_time_created[0]

    # initialize list of time differences
    all_time_diffs = list()

    # compute time differences
    for time_destination in all_time_created[1:]:
        diff = computeTimeDifference(time_origin, time_destination)
        all_time_diffs.append(diff.seconds)

    # convert time differences to torch tensor
    edge_attr = torch.tensor(np.array(all_time_diffs).reshape((len(all_time_diffs), 1)), dtype=torch.float)

    # add edge attributes to data object
    cascade = Data(x=cascade.x, edge_index=cascade.edge_index, edge_attr=edge_attr, y=cascade.y)

    return cascade


def addAgeUntilTweet(cascade):
    # COMPUTE AGE OF USER ACCOUNT UNTIL TWEET

    x = list()  # initialize new list of features

    for features in cascade.x:
        time_tweet = features[0]  # extract raw tweet creation date
        time_user = features[-1]  # extract raw user creation date
        time_diff = computeTimeDifference(time_tweet, time_user).days  # compute time difference
        features = np.delete(features, [0, -1])  # remove time tweet creation and time user creation
        x.append(np.concatenate((features, np.array(time_diff, ndmin=1))))  # add to new list of features

    x = torch.tensor(x, dtype=torch.float)  # convert to tensor
    cascade = Data(x=x, edge_index=cascade.edge_index, edge_attr=cascade.edge_attr,
                   y=cascade.y)  # crate new cascade representation
    return cascade


def normalize(value, mu, sigma):
        return (value - mu) / sigma


def normalizeCascade(cascade, Means, Sigmas):
    edge_index = cascade.edge_index
    edge_attr = cascade.edge_attr
    y = cascade.y.to(torch.long)

    # normalize node features
    for node in cascade.x:  # loop over each node in the cascade

        # initialize new feature vector
        normalized_nodefeatures = list()

        # normalize each feature with its corresponding mean and sigma
        normalized_nodefeatures.append(normalize(node[0], Means['TextEmb'], Sigmas['TextEmb']))
        normalized_nodefeatures.append(normalize(node[1], Means['HashEmb'], Sigmas['HashEmb']))
        normalized_nodefeatures.append(normalize(node[2], Means['NameEmb'], Sigmas['NameEmb']))
        normalized_nodefeatures.append(normalize(node[3], Means['DescEmb'], Sigmas['DescEmb']))
        normalized_nodefeatures.append(normalize(node[4], Means['Followers'], Sigmas['Followers']))
        normalized_nodefeatures.append(normalize(node[5], Means['Following'], Sigmas['Following']))
        normalized_nodefeatures.append(normalize(node[6], Means['Listed'], Sigmas['Listed']))
        normalized_nodefeatures.append(normalize(node[7], Means['Age'], Sigmas['Age']))


    # normalize edge features
    normalized_edgefeatures = [normalize(time, Means['TimeDiff'], Sigmas['TimeDiff']) for time in edge_attr]

    x = torch.tensor(normalized_nodefeatures, dtype=torch.float)   # convert new feature list to torch tensor
    normalized_edgefeatures = torch.tensor(  # convert normalized edge feature to torch tensor
        np.array(normalized_edgefeatures), dtype=torch.float
    )

    # create new cascade representation
    cascade = Data(x=normalized_nodefeatures, edge_index=edge_index, edge_attr=normalized_edgefeatures, y=y)
    return cascade


def normalizeFeatures(data):
    # initialize total feature list
    TextEmb = list()
    HashEmb = list()
    NameEmb = list()
    DescEmb = list()
    Followers = list()
    Following = list()
    Listed = list()
    Age = list()
    TimeDiff = list()

    # initialize new graph dataset
    graph_dataset = list()

    print()
    print('====== Start extracting node features ======')
    for cascade in tqdm(data):  # loop over each cascade in data
        newcascade = addTimeBetweenTweets(cascade)  # add edge feature
        newcascade = addAgeUntilTweet(newcascade)  # add user account age
        graph_dataset.append(newcascade)  # add to new graph data set
        TimeDiff.extend(newcascade.edge_attr)  # extract all edge features
        for features in newcascade.x:  # extract features of all nodes in newcascade
            TextEmb.extend([features[0]])  # extract text embedding
            HashEmb.extend([features[1]])  # extract hashtag embedding
            NameEmb.extend([features[2]])  # extract name embedding
            DescEmb.extend([features[3]])  # extract description embedding
            Followers.extend([features[4]])  # extract follower count
            Following.extend([features[5]])  # extract following count
            Listed.extend([features[6]])  # extract listed count
            Age.extend([features[7]])  # extract age until tweet

    # store all means
    Means = dict()
    Means['TextEmb'] = np.mean(TextEmb)
    Means['HashEmb'] = np.mean(HashEmb)
    Means['NameEmb'] = np.mean(NameEmb)
    Means['DescEmb'] = np.mean(DescEmb)
    Means['Followers'] = np.mean(Followers)
    Means['Following'] = np.mean(Following)
    Means['Listed'] = np.mean(Listed)
    Means['Age'] = np.mean(Listed)
    Means['TimeDiff'] = np.mean(TimeDiff)

    print()
    print('====== Mean values of node features ======')
    for k,v in Means.items():
        print(k,v)
    print()

    # store all standard deviations
    Sigmas = dict()
    Sigmas['TextEmb'] = np.std(TextEmb)
    Sigmas['HashEmb'] = np.std(HashEmb)
    Sigmas['NameEmb'] = np.std(NameEmb)
    Sigmas['DescEmb'] = np.std(DescEmb)
    Sigmas['Followers'] = np.std(Followers)
    Sigmas['Following'] = np.std(Following)
    Sigmas['Listed'] = np.std(Listed)
    Sigmas['Age'] = np.std(Listed)
    Sigmas['TimeDiff'] = np.std(TimeDiff)

    print()
    print('====== Standard deviations of node features ======')
    for k,v in Sigmas.items():
        print(k,v)
    print()


    # normalize each cascade and add to new dataset
    final_graph_dataset = list()

    print()
    print('====== Start normalizing node features ======')
    print()

    for cascade in tqdm(graph_dataset):
        final_graph_dataset.append(normalizeCascade(cascade, Means, Sigmas))

    print()
    print('====== Example of standardized cascade ======')
    example = final_graph_dataset[0]
    print('Cascade object: ', example)
    print('Feature matrix: ', example.x)
    print('Edge features: ', example.edge_attr)
    print('Label: ', example.y)
    print()


    return final_graph_dataset
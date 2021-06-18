"""
Functions for preprocessing the twitter data.
"""

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data


# list of features
# 0 location
# 1 language -> almost always None
# 2 profile description
# 3 account creation
# 4 verified
# 5 favorites count
# 6 listed count
# 7 status count
# 8 followers count
# 9 friends count
# 10 retweet time
# 11 source device
# 12 retweet count
# 13 favorites count
# 14 text
# 15 hashtags
# 16 interaction strength


def change_to_baseline(cascade):
    N = len(cascade.x) - 1   # root id and no. edges
    origin = np.repeat(N, N)
    destination = np.arange(N)
    indices = torch.tensor(np.array((origin, destination)), dtype=torch.long)
    return Data(x=cascade.x, edge_index=indices, y=cascade.y)


def normalize(value, mu, sigma):
        return (value - mu) / sigma


def normalizeCascade(cascade, Means, Sigmas, as_baseline):

    # change label type to Long
    y = cascade.y.to(torch.long)

    # initialize new feature matrix
    x = list()

    # normalize node features
    for feature in cascade.x:  # loop over each node in the cascade

        # initialize new feature vector
        normalized_nodefeatures = list()

        # normalize each feature with its corresponding mean and sigma
        normalized_nodefeatures.append(normalize(feature[0], Means['location'], Sigmas['location']))
        normalized_nodefeatures.append(normalize(feature[1], Means['description'], Sigmas['description']))
        normalized_nodefeatures.append(normalize(feature[2], Means['account_created_at'], Sigmas['account_created_at']))
        normalized_nodefeatures.append(normalize(feature[3], Means['verified'], Sigmas['verified']))
        normalized_nodefeatures.append(normalize(feature[4], Means['favorites_count'], Sigmas['favorites_count']))
        normalized_nodefeatures.append(normalize(feature[5], Means['listed_count'], Sigmas['listed_count']))
        normalized_nodefeatures.append(normalize(feature[6], Means['statuses_count'], Sigmas['statuses_count']))
        normalized_nodefeatures.append(normalize(feature[7], Means['followers_count'], Sigmas['followers_count']))
        normalized_nodefeatures.append(normalize(feature[8], Means['friends_count'], Sigmas['friends_count']))
        normalized_nodefeatures.append(normalize(feature[9], Means['tweet_created_at'], Sigmas['tweet_created_at']))
        normalized_nodefeatures.append(normalize(feature[10], Means['source'], Sigmas['source']))
        normalized_nodefeatures.append(normalize(feature[11], Means['retweet_count'], Sigmas['retweet_count']))
        # normalized_nodefeatures.append(normalize(feature[12], Means['favorite_count'], Sigmas['favorite_count']))
        normalized_nodefeatures.append(normalize(feature[13], Means['text'], Sigmas['text']))
        normalized_nodefeatures.append(normalize(feature[14], Means['hashtag'], Sigmas['hashtag']))

        if not as_baseline:
            normalized_nodefeatures.append(normalize(feature[15], Means['is_value'], Sigmas['is_value']))

        # add to new feature matrix
        x.append(normalized_nodefeatures)

    # convert new feature matrix to torch tensor
    x = torch.tensor(np.array(x), dtype=torch.float)

    # return new cascade representation 
    return Data(x=x, edge_index=cascade.edge_index, y=y)


def normalizeFeatures(data, as_baseline=False):

    # initialize lists of all features
    location = list()
    description = list()
    account_created_at = list()
    verified = list()
    favorites_count = list()
    listed_count = list()
    statuses_count = list()
    followers_count = list()
    friends_count = list()
    tweet_created_at = list()
    source = list()
    retweet_count = list()
    # favorite_count = list()
    text = list()
    hashtag = list()
    is_value = list()


    # initialize total feature dict
    all_features = dict()

    print()
    print('====== Start extracting node features ======')
    for cascade in tqdm(data):  # loop over each cascade in data
        for feature in cascade.x:  # extract features of all nodes
            location.extend([feature[0]])
            description.extend([feature[1]])
            account_created_at.extend([feature[2]])
            verified.extend([feature[3]])
            favorites_count.extend([feature[4]])
            listed_count.extend([feature[5]])
            statuses_count.extend([feature[6]])
            followers_count.extend([feature[7]])
            friends_count.extend([feature[8]])
            tweet_created_at.extend([feature[9]])
            source.extend([feature[10]])
            retweet_count.extend([feature[11]])
            # favorite_count.extend([feature[12]])
            text.extend([feature[13]])
            hashtag.extend([feature[14]])

            if not as_baseline:
                is_value.extend([feature[15]])


    # store all means
    Means = dict()
    Means['location'] = np.mean(location)
    Means['description'] = np.mean(description)
    Means['account_created_at'] = np.mean(account_created_at)
    Means['verified'] = np.mean(verified)
    Means['favorites_count'] = np.mean(favorites_count)
    Means['listed_count'] = np.mean(listed_count)
    Means['statuses_count'] = np.mean(statuses_count)
    Means['followers_count'] = np.mean(followers_count)
    Means['friends_count'] = np.mean(friends_count)
    Means['tweet_created_at'] = np.mean(tweet_created_at)
    Means['source'] = np.mean(source)
    Means['retweet_count'] = np.mean(retweet_count)
    # Means['favorite_count'] = np.mean(favorite_count)
    Means['text'] = np.mean(text)
    Means['hashtag'] = np.mean(hashtag)

    if not as_baseline:
        Means['is_value'] = np.mean(is_value)

    print()
    print('====== Mean values of node features ======')
    for k,v in Means.items():
        print(k, v)
    print()

    # store all standard deviations
    Sigmas = dict()
    Sigmas['location'] = np.std(location)
    Sigmas['description'] = np.std(description)
    Sigmas['account_created_at'] = np.std(account_created_at)
    Sigmas['verified'] = np.std(verified)
    Sigmas['favorites_count'] = np.std(favorites_count)
    Sigmas['listed_count'] = np.std(listed_count)
    Sigmas['statuses_count'] = np.std(statuses_count)
    Sigmas['followers_count'] = np.std(followers_count)
    Sigmas['friends_count'] = np.std(friends_count)
    Sigmas['tweet_created_at'] = np.std(tweet_created_at)
    Sigmas['source'] = np.std(source)
    Sigmas['retweet_count'] = np.std(retweet_count)
    # Sigmas['favorite_count'] = np.std(favorite_count)
    Sigmas['text'] = np.std(text)
    Sigmas['hashtag'] = np.std(hashtag)

    if not as_baseline:
        Sigmas['is_value'] = np.std(is_value)

    print()
    print('====== Standard deviations of node features ======')
    for k,v in Sigmas.items():
        print(k, v)
    print()


    print()
    print('====== Start normalizing node features ======')
    print()

    # normalize each cascade and add to new dataset
    final_graph_dataset = list()
    for cascade in tqdm(data):
        if as_baseline:
            cascade = change_to_baseline(cascade)
            final_graph_dataset.append(normalizeCascade(cascade, Means, Sigmas, as_baseline))
        else:
            final_graph_dataset.append(normalizeCascade(cascade, Means, Sigmas, as_baseline))

    print()
    print('====== Example of standardized cascade ======')
    example = final_graph_dataset[0]
    print('Cascade object: ', example)
    print('Edge indices:', example.edge_index)
    print('Feature matrix: ', example.x)
    print('Edge features: ', example.edge_attr)
    print('Label: ', example.y)
    print()

    return final_graph_dataset


if __name__ == '__main__':
    pass



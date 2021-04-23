import os
import json
import numpy as np
import time
import argparse
from tqdm import tqdm

import torch
from torch_geometric.data import Data


# initialize arguments
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='/data/s1805819/fakenewsnet_dataset/politifact',
                    help='enter your Twitter data path')

parser.add_argument('--glove', type=str,  default='/data/s2583550/FakeNewsDetection/CreateData/resources/glove.twitter.27B.200d.txt',
                    help='enter your GloVe data path')

parser.add_argument('--target_folder', type=str, default='/data/s2583550/FakeNewsDetection',
                    help='enter your target data path')

args = parser.parse_args()
print(args)


class GloVe:
    def __init__(self, debug=False):
        print(f"setting up GloVe dict")
        self.embeddings_dict = {}

        if not debug:
            # args.glove = data path of GloVe data, to be entered in the beginning
            filename = args.glove
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    self.embeddings_dict[word] = vector

    # For tweet content and user description embeddings we averaged together the embeddings of the
    # constituent words(GloVe[27] 200 - dimensional vectors pre - trained on Twitter).
    def get_average_embedding(self, word_array):
        embeddings = []
        for word in word_array.split(" "):
            if word in self.embeddings_dict.keys():
                print(f'word is in embedding: {word}')
                embeddings.append(self.embeddings_dict[word])
        if len(embeddings) == 0:
            print('embeddings == 0')
            return 0
        return np.mean(embeddings)


def _parse_twitter_time(timestamp):
    return np.compat.long(
        time.strftime(
            "%Y%m%d%H%M%S", time.strptime(timestamp, "%a %b %d %H:%M:%S +0000 %Y")
        )
    )


class CreateCascade:
    def __init__(self, _data_folder, _news_id, glove, label):
        self.cascade_threshold = 5
        self.GloVe = glove
        self.i = 0
        self.data_folder = _data_folder
        self.news_id = _news_id
        self.cascade_ids = self._load_cascades()
        self.label = label

    def _extract_tweet_information(self, tweet_object):
        tweet_info = []
        tweet_info.append(_parse_twitter_time(tweet_object["created_at"]))
        tweet_info.append(self.GloVe.get_average_embedding(tweet_object["text"]))
        tweet_info.append(
            self.GloVe.get_average_embedding(
                " ".join([i["text"] for i in tweet_object["entities"]["hashtags"]])
            )
        )
        tweet_info.append(
            self.GloVe.get_average_embedding(tweet_object["user"]["name"])
        )
        tweet_info.append(
            self.GloVe.get_average_embedding(tweet_object["user"]["description"])
        )
        tweet_info.append(tweet_object["user"]["followers_count"])
        tweet_info.append(tweet_object["user"]["friends_count"])
        tweet_info.append(tweet_object["user"]["listed_count"])
        tweet_info.append(_parse_twitter_time(tweet_object["user"]["created_at"]))
        return tweet_info

    def _get_tweet_data(self):
        tweet_file = os.path.join(
            self.data_folder, self.news_id, "tweets", self.cascade_ids[self.i]
        )
        if os.path.exists(tweet_file):
            tweet_data = open(file=tweet_file)
            tweet_json = json.load(tweet_data)
            return tweet_json
        else:
            raise FileNotFoundError

    def _get_cascade_data(self):
        retweet_file = os.path.join(
            self.data_folder, self.news_id, "retweets", self.cascade_ids[self.i]
        )
        if os.path.exists(retweet_file):
            retweet_data = open(file=retweet_file)
            retweet_json = json.load(retweet_data)
            retweets = retweet_json["retweets"]
            if len(retweets) < self.cascade_threshold:
                #TODO: change to logging
                # print(
                #     f"cascade does not meet the threshold of {self.cascade_threshold}"
                # )
                return None
            else:
                #TODO: change to logging
                # print(
                #     f"\tfound {len(retweets)} for cascade {self.i} of news item {self.news_id}"
                # )
                return retweets
        else:
            raise FileNotFoundError

    def _load_cascades(self):
        tweets_folder = os.path.join(self.data_folder, self.news_id, "tweets")
        if os.path.exists(tweets_folder):
            tweets = os.listdir(tweets_folder)
            #TODO: change to logging
            # print(f"found {len(tweets)} retweets cascades for news id {self.news_id}")
            return tweets
        else:
            #TODO: change to logging
            # print(f"found no tweets for news id {self.news_id}")
            return []

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.cascade_ids):
            raise StopIteration

        try:
            # collect data
            current_tweet_data = self._get_tweet_data()
            current_cascade_data = self._get_cascade_data()
        except FileNotFoundError:
            current_tweet_data = None
            current_cascade_data = None
        except json.JSONDecodeError:
            #TODO: change to logging
            # print('WARNING: it seems the json file is corrupted')
            current_tweet_data = None
            current_cascade_data = None

        # remove item if threshold is not reached
        if current_cascade_data is None:
            del self.cascade_ids[self.i]
            if self.i >= len(self.cascade_ids):
                raise StopIteration
            return None

        # create cascade (NOTE: now in simple form)

        # create x array ->  data.x: Node feature matrix with shape [num_nodes, num_node_features]
        # create edge_index -> Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        x = []
        edge_index = []
        x.append(self._extract_tweet_information(current_tweet_data))
        for indx, rt in enumerate(current_cascade_data):
            # from -> to
            edge_index.append([0, indx])
            # add features
            x.append(self._extract_tweet_information(rt))
        x_tensor = torch.tensor(data=x)
        edge_index_tensor = torch.tensor(data=edge_index, dtype=torch.long)
        y_tensor = torch.tensor(data=self.label, dtype=torch.int)

        # converting to data object
        data = Data(
            x=x_tensor, edge_index=edge_index_tensor.t().contiguous(), y=y_tensor
        )

        # print(data)
        # finish iteration
        self.i += 1
        return data, self.cascade_ids[self.i - 1]


if __name__ == "__main__":
    print(f"started the creation of novel cascades")


    def _get_news_ids(data_folder):
        folders = os.listdir(data_folder)
        return folders

    # When Debug=True GloVe always returns 0 :)
    glove = GloVe(debug=False)
    # args.dataset = data path of twitter data, to be entered in the beginning
    root = args.dataset
    # root = os.path.join("D:", "Onderzoek", "FakeNews", "fakenewsnet_dataset", "politifact")

    for is_true in [False, True]:
        label = "real" if is_true else "fake"
        news_ids = _get_news_ids(os.path.join(root, label))
        for news_id in tqdm(news_ids, desc=f'creating cascades of {label} news items'):
            cascade_object = CreateCascade(os.path.join(root, label), news_id, glove, is_true)
            for cascade in cascade_object:
                if cascade is not None:
                    # print(f"\t\t{cascade[0]}")
                    filename = os.path.join(
                        # args.target_folder = target folder of news cascades, to be entered in the beginning
                        args.target_folder, "cascades", f'{news_id}-{cascade[1].split(".")[0]}.pt'
                    )
                    torch.save(cascade[0], filename)

import os
from collectDataNew import RetrieveData
from graphviz import Digraph
from tqdm import tqdm
import pickle
import json
import numpy as np
import time
import random

# libs for creating the torch based network
import torch
from torch_geometric.data import Data

class GloVe:
    def __init__(self, debug=True):
        print(f"setting up GloVe dict")
        self.embeddings_dict = {}

        if not debug:
            filename = os.path.join("/data", "s1805819", "glove", "glove.twitter.27B.200d.txt")
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
                embeddings.append(self.embeddings_dict[word])
        if len(embeddings) == 0:
            return 0
        return np.mean(embeddings)


def _parse_twitter_time(timestamp):
    return np.compat.long(
        time.strftime(
            "%Y%m%d%H%M%S", time.strptime(timestamp, "%a %b %d %H:%M:%S +0000 %Y")
        )
    )

def _parse_twitter_source(source):
    source_link = source.split('"')[1]
    if source_link == "http://twitter.com":
        return "twitter"
    elif source_link == "https://mobile.twitter.com":
        return "mobile"
    else:
        return source_link.split('/')[-1].replace("www.", "").split(".")[0]

class CreateCascade:
    def __init__(self, _data: RetrieveData, glove, label):
        self.data = _data
        self.glove = glove
        self.label = label
        self.weights = [1, 1, 1]



    def user_features(self, user_id, IS_value):
        tweet_object = self.data.get_tweet_object(user_id)
        user_profile = tweet_object["user"]

        # User profile (geolocalization and profile settings, language, word embedding of
        # user profile self-description, date of account creation, and whether it has been verified)
        location = self.glove.get_average_embedding(user_profile["location"])
        language = user_profile["lang"] # almost always None
        profile_description = self.glove.get_average_embedding(user_profile["description"])
        account_creation = _parse_twitter_time(user_profile["created_at"])
        verified = 1 if user_profile["verified"] else 0

        # User activity (number of favorites, lists, and statuses)
        favorites_count = user_profile["favourites_count"]
        listed_count = user_profile["listed_count"]
        status_count = user_profile["statuses_count"]

        # number of followers and friends
        followers = user_profile["followers_count"]
        friends = user_profile["friends_count"]

        # tweet features
        retweet_time = _parse_twitter_time(tweet_object["created_at"])
        source_device = self.glove.get_average_embedding(_parse_twitter_source(tweet_object["source"]))

        # original tweet
        retweets = tweet_object["retweet_count"]
        favorites = tweet_object["favorite_count"]
        text = self.glove.get_average_embedding(tweet_object["text"])
        hashtags = self.glove.get_average_embedding(' '.join([h['text'] for h in tweet_object["entities"]["hashtags"]]))

        interaction_strength = IS_value

        return [
            location,
            # language, this is always None
            profile_description,
            account_creation,
            verified,
            favorites_count,
            listed_count,
            status_count,
            followers,
            friends,
            retweet_time,
            source_device,
            retweets,
            favorites,
            text,
            hashtags,
            interaction_strength
        ]


    def get_cascades(self):
        # print(f'# of retweets {len(self.data.retweets)}')

        # check ordering
        for retweet_index in range(0, len(self.data.retweets)-1):
            assert self.data.retweets[retweet_index]["id"] > self.data.retweets[retweet_index+1]["id"]

        orderd_ids = [i["user"]["id"] for i in self.data.retweets]
        orderd_ids.append(self.data.original_tweet_object["user"]["id"])

        interaction_dict = {}
        # create connection dict
        for retweet_index in range(0, len(orderd_ids)):
            current_user = orderd_ids[retweet_index]
            interaction_dict[current_user] = {}
            for retweets_before in range(retweet_index+1, len(orderd_ids)):
                user_to_check = orderd_ids[retweets_before]

                interaction = self.data.user_timelines[current_user]

                replies = [int(i) for i in interaction['replies']].count(user_to_check)
                retweets = [int(i) for i in interaction['retweets']].count(user_to_check)
                quotes = [int(i) for i in interaction['quotes']].count(user_to_check)
                if replies != 0 or retweets != 0 or quotes != 0:
                    interaction_dict[current_user][user_to_check] = \
                        self.weights[0] * replies +\
                        self.weights[1] * quotes +\
                        self.weights[2] * retweets

        # create x array ->  data.x: Node feature matrix with shape [num_nodes, num_node_features]
        # create edge_index -> Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        x = []
        edge_index = []

        root_user = list(interaction_dict.keys())[-1]
        for u1 in [i for i in interaction_dict.keys()][:-1]:
            # determine link and max IS value
            max_value = 0
            highest_IS_user = 0
            if len(interaction_dict[u1]) > 0:
                for u2 in interaction_dict[u1].keys():
                    v2 = interaction_dict[u1][u2]
                    if v2 > max_value:
                        max_value = v2
                        highest_IS_user = u2

            # add user to node feature vector
            x.append(self.user_features(u1, max_value))

            # add connection to edge list
            from_user_id = highest_IS_user
            to_user = list(interaction_dict.keys()).index(u1)
            if from_user_id == 0: # no connection found
                # from -> to
                root_user_index = list(interaction_dict.keys()).index(root_user)
                edge_index.append([root_user_index, to_user])
            else:
                from_user = list(interaction_dict.keys()).index(from_user_id)
                edge_index.append([from_user, to_user])

        # adding the root at the back
        x.append(self.user_features(root_user, -1))

        # create tensor
        x_tensor = torch.tensor(data=x)
        edge_index_tensor = torch.tensor(data=edge_index, dtype=torch.long)
        y_tensor = torch.tensor(data=self.label, dtype=torch.int)

        # converting to data object
        data = Data(
            x=x_tensor, edge_index=edge_index_tensor.t().contiguous(), y=y_tensor
        )
        return data

def create_plot(cascade, filename):
    dot = Digraph(comment="cascade")
    for unlinked_user in cascade["users_to_craw"]:
        dot.node(str(unlinked_user))
    linked_users = list(cascade.keys())[2:]
    for linked_user in linked_users:
        dot.edge(str(cascade[linked_user]["maxFrom"]["user"]), str(linked_user))
    dot.render(os.path.join("output", str(filename)), view=False)


def grab_all_cascades(data_root):
    true_cascades = []
    fake_cascades = []
    for type_of_story in ["real", "fake"]:
        story_path = os.path.join(data_root, "politifact", type_of_story)
        stories = os.listdir(story_path)
        for story in tqdm(stories, desc="number of stories processed"):
            cascade_roots_path = os.path.join(story_path, story, "tweets")
            if os.path.exists(cascade_roots_path):
                cascade_roots = os.listdir(cascade_roots_path)
                for cascade_root in cascade_roots:
                    retweets_path = os.path.join(
                        story_path, story, "retweets", cascade_root
                    )
                    tweet_path = os.path.join(story_path, story, "tweets", cascade_root)
                    if os.path.exists(retweets_path) and os.path.exists(tweet_path):
                        retweets_object = json.load(open(retweets_path, "rb"))
                        if len(retweets_object["retweets"]) >= 5:
                            if type_of_story == "real":
                                true_cascades.append([tweet_path, retweets_path])
                            else:
                                fake_cascades.append([tweet_path, retweets_path])

def main():
    # # randomly sampling 1000 true and 1000 fake news cascades
    # true_cascades = pickle.load(open("resources/true_cascades.p", "rb"))
    # fake_cascades = pickle.load(open("resources/fake_cascades.p", "rb"))
    # random_sample_true = random.sample(true_cascades, 1000)
    # pickle.dump(random_sample_true, open("resources/randomly_sampled_true.p", "wb"))
    #
    # random_sample_fake = random.sample(fake_cascades, 1000)
    # pickle.dump(random_sample_fake, open("resources/randomly_sampled_fake.p", "wb"))

    true_cascades = pickle.load(open("resources/randomly_sampled_true.p", "rb"))
    fake_cascades = pickle.load(open("resources/randomly_sampled_fake.p", "rb"))

    # TODO: this should be changed to where you have the data, if you run it in a liacs DS machine you can use
    data_root = os.path.join("/data", "s1805819", "fakenewsnet_dataset")
    # data_root = os.path.join("D:", "Onderzoek", "FakeNews", "fakenewsnet_dataset")
    user_timeline_root = os.path.join(data_root, "user_timeline_tweets_new")

    glove = GloVe(debug=False)
    for type_of_story in ['true', 'fake']:
        for cascade_files in tqdm(
                true_cascades if type_of_story == 'true' else fake_cascades,
                desc=f"creating {type_of_story} news cascades"
        ):
            tweet_path = cascade_files[0]
            tweet_id_file = tweet_path.split("\\")[-1]
            story = tweet_path.split("\\")[-3]

            tweet_file = os.path.join(data_root, "politifact", "real" if type_of_story == 'true' else "fake", story,
                                      "tweets", tweet_id_file)
            retweets_file = os.path.join(data_root, "politifact", "real" if type_of_story == 'true' else "fake", story,
                                      "retweets", tweet_id_file)

            retweets_object = json.load(open(retweets_file, "rb"))
            original_tweet_object = json.load(open(tweet_file, "rb"))
            user_timeline_folder = os.path.join(user_timeline_root, story)
            data_object = RetrieveData(
                user_timeline_folder=user_timeline_folder,
                original_tweet_object=original_tweet_object,
                retweets_object=retweets_object,
                simple=False
            )
            cascade = CreateCascade(data_object, glove, 1 if type_of_story == 'true' else 0)
            cascade_object = cascade.get_cascades()
            torch.save(cascade_object, os.path.join(data_root, 'cascades', f'{story}-{original_tweet_object["id"]}.pt'))
            print(cascade_object)
        print(f"found {len(true_cascades)} correct {type_of_story} cascades")


if __name__ == "__main__":
    main()
"""
Code taken from https://github.com/paolazola/Interaction-strength-analysis-to-model-retweet-cascade-graphs
The code is adjusted for out needs, however, no changes to the calculations are made
"""


# ------------------------------------------------------------------------------
# IS-BASED RETWEETS CASCADES
# ------------------------------------------------------------------------------

import os
from CreateData.collectData import RetrieveData
import pandas as pd
import numpy as np
from datetime import datetime
from graphviz import Digraph
import tempfile
import datetime as dt
import time
from tqdm import tqdm
import json
import itertools
import multiprocessing as mp
from functools import partial


class CreateCascade:
    def __init__(self, data):

        self.data = data

        self.RETWEET = 0
        self.QUOTE = 1
        self.COMMENT = 2

        self.weights = []
        self.weights.append((0.35, 1, 0.7))

        self.users_to_check_404 = []
        self.last = True

        # TODO: this is not relevant anymore, I think
        max_history = 9600

        self.root_tweet = self.data.original_tweet_object
        self.root_tweetID = self.root_tweet["id"]
        self.root_date = self.root_tweet["created_at"]

    def get_cascades(self):
        all_link, tweetsinfo = self.links()
        cascades = self.levels(self.root_tweetID)
        cascades = self.links_levels_parallelo(cascades, tweetsinfo, all_link)
        return cascades[0]

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # networks links probabilities
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    def f_all_links(self, all_links, users_id_unique, result):
        users_link = {}
        index, value = all_links
        for u in users_id_unique:
            if len(result.loc[u]):
                dd = pd.DataFrame(result.loc[u]).T
                if len(dd) < 2:
                    dd.index = dd.to
                    del (dd["to"], dd["from"])
                    dd["probabilities"] = 1
                    users_link[u] = dd
                else:
                    dd = pd.DataFrame(result.loc[u]).groupby("to").sum()
                    dd["probabilities"] = dd[index] / dd[index].sum()
                    users_link[u] = dd
        return users_link

    def f_links(self, cascades, tweetsinfo, all_users_link, retwettatore, k):
        index, value = cascades
        if retwettatore not in list(value.keys()):
            value = self.habitual_friend(
                value, retwettatore, k, all_users_link[index], index, tweetsinfo
            )
        return value

    def parallel_runs(self, all_links, users_id_unique, result):
        par_work = partial(
            self.f_all_links, users_id_unique=users_id_unique, result=result
        )
        res = pool.map(par_work, all_links)
        return res

    def links_parallel_runs(
        self, cascades, tweetsinfo, all_users_link, retwettatore, k
    ):
        par_work = partial(
            self.f_links,
            tweetsinfo=tweetsinfo,
            all_users_link=all_users_link,
            retwettatore=retwettatore,
            k=k,
        )
        cascades = pool.map(par_work, cascades)
        return cascades

    def links(self):
        all_info = {}
        all_info = self.data.get_retweet_list()
        all_info["results"].append(
            {
                "content_id": str(self.root_tweet["id"]),
                "date": self.root_tweet["created_at"],
                "user_id": self.root_tweet["user"]["id"],
            }
        )
        tweetsinfo = pd.DataFrame(
            {
                "user_id": [l["user_id"] for l in all_info["results"]],
                "timestamp": [
                    datetime.strptime(l["date"], "%a %b %d %H:%M:%S +0000 %Y")
                    for l in all_info["results"]
                ],
            }
        )

        tweetsinfo = tweetsinfo.drop_duplicates("user_id")
        tweetsinfo = tweetsinfo.reset_index()

        # CLEAN
        del all_info

        users_info_df = pd.DataFrame(
            index=tweetsinfo["user_id"].drop_duplicates(),
            columns=["#friends", "#followers", "favourite", "status"],
        )

        usersIDs_list = list(users_info_df.index)

        for i in tqdm(range(0, len(tweetsinfo))):
            userID = tweetsinfo["user_id"].loc[i]
            is_call_ok = False
            user_info = {}
            while not is_call_ok:
                try:
                    user_info = self.data.get_users_metrics(userID)
                    is_call_ok = True
                except:
                    is_call_ok = False

            if user_info["status"] == 200:
                users_info_df["#friends"].iloc[i] = user_info["friends_count"]
                users_info_df["#followers"].iloc[i] = user_info["followers_count"]
                users_info_df["favourite"].iloc[i] = user_info["favourites_count"]
                users_info_df["status"].iloc[i] = user_info["statuses_count"]

        autore_ret, autore_originale_ret, count_ret = self.data.get_retweet_matrix(
            usersIDs_list
        )
        autore_quot, autore_originale_quot, count_quot = self.data.get_quote_matrix(
            usersIDs_list
        )
        autore_comm, autore_originale_comm, count_comm = self.data.get_reply_matrix(
            usersIDs_list
        )
        weighted_ret = []
        weighted_quot = []
        weighted_comm = []

        for c in range(0, len(self.weights)):
            weighted_temp_ret = []
            weighted_temp_quot = []
            weighted_temp_comm = []
            for i in range(0, max(len(autore_ret), len(autore_quot), len(autore_comm))):
                if i < len(autore_ret):
                    temp = users_info_df["#followers"].loc[autore_ret[i]]
                    if temp == 0:
                        temp = 0.1
                    weighted_temp_ret.append(
                        (count_ret[i] * self.weights[c][self.RETWEET]) / temp
                    )
                if i < len(autore_quot):
                    temp = users_info_df["#followers"].loc[autore_quot[i]]
                    if temp == 0:
                        temp = 0.1
                    weighted_temp_quot.append(
                        (count_quot[i] * self.weights[c][self.QUOTE]) / temp
                    )
                if i < len(autore_comm):
                    temp = users_info_df["#followers"].loc[autore_comm[i]]
                    if temp == 0:
                        temp = 0.1
                    weighted_temp_comm.append(
                        (count_comm[i] * self.weights[c][self.COMMENT]) / temp
                    )
            weighted_ret.append(weighted_temp_ret)
            weighted_quot.append(weighted_temp_quot)
            weighted_comm.append(weighted_temp_comm)

        ret_df = pd.DataFrame(
            {"from": autore_ret, "to": autore_originale_ret, "count": count_ret}
        ).join(pd.DataFrame(weighted_ret).T)
        ret_df.index = ret_df["from"]
        quotes_df = pd.DataFrame(
            {"from": autore_quot, "to": autore_originale_quot, "count": count_quot}
        ).join(pd.DataFrame(weighted_quot).T)
        quotes_df.index = quotes_df["from"]
        comments_df = pd.DataFrame(
            {"from": autore_comm, "to": autore_originale_comm, "count": count_comm}
        ).join(pd.DataFrame(weighted_comm).T)
        comments_df.index = comments_df["from"]
        del autore_ret
        del autore_quot
        del autore_comm
        del autore_originale_ret
        del autore_originale_quot
        del autore_originale_comm
        del count_ret
        del count_quot
        del count_comm
        del weighted_ret
        del weighted_quot
        del weighted_comm

        frames = [ret_df, quotes_df, comments_df]
        result = pd.concat(frames)

        all_links = [{} for i in range(0, len(self.weights))]

        users_id_unique = list(set(result.index))

        if __name__ == "__main__":
            all_links = self.parallel_runs(
                enumerate(all_links), users_id_unique, result
            )
            del frames
            del users_id_unique
            del result

        return all_links, tweetsinfo

    def findFriends(
        self,
        retweet_matrix,
        quotes_matrix,
        comments_matrix,
        users_info_df,
        weight_retweet,
        weight_quotes,
        weight_comments,
    ):
        users_link = {}
        for user in users_info_df.index:
            try:
                keep_ret = retweet_matrix.loc[user].iloc[
                    retweet_matrix.loc[user].nonzero()[0]
                ]
                try:
                    keep_ret = keep_ret.drop([user], axis=0)
                except KeyError:
                    pass
                keep_quotes = quotes_matrix.loc[user].iloc[
                    quotes_matrix.loc[user].nonzero()[0]
                ]
                try:
                    keep_quotes = keep_quotes.drop([user], axis=0)
                except KeyError:
                    pass

                keep_comm = comments_matrix.loc[user].iloc[
                    comments_matrix.loc[user].nonzero()[0]
                ]
                try:
                    keep_comm = keep_comm.drop([user], axis=0)
                except KeyError:
                    pass
                retweet_user = [
                    (
                        c,
                        (weight_retweet * keep_ret[c])
                        / users_info_df["#followers"].loc[c],
                    )
                    for c in keep_ret.index
                ]
                quotes_user = [
                    (
                        c,
                        (weight_quotes * keep_quotes[c])
                        / users_info_df["#followers"].loc[c],
                    )
                    for c in keep_quotes.index
                ]
                comm_user = [
                    (
                        c,
                        (weight_comments * keep_comm[c])
                        / users_info_df["#followers"].loc[c],
                    )
                    for c in keep_comm.index
                ]

            except IndexError:
                pass

            junto = pd.DataFrame(retweet_user + quotes_user + comm_user)
            if len(junto) != 0:
                junto.columns = ["userID", "value"]
                final = junto.groupby(junto["userID"]).sum()
                final["probabilities"] = final["value"] / final["value"].sum()

                users_link[user] = final
        return users_link

    # ------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    #          CASCADE CREATION
    # ------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    # livello zero:
    def sort_by_time(self, tweetsinfo):
        tweetsinfo = tweetsinfo.sort_values(by=["timestamp"])

    def levels(self, root_tweetID):
        # info del tweet
        root_timestamp = datetime.strptime(self.root_date, "%a %b %d %H:%M:%S +0000 %Y")
        root_userID = self.root_tweet["user"]["id"]
        cascades = []
        for c in range(0, len(self.weights)):
            cascades.append(
                {
                    "users_to_craw": [],
                    "cascade_params": {
                        "rtw_weight": self.weights[c][self.RETWEET],
                        "qtd_weight": self.weights[c][self.QUOTE],
                        "comm_weight": self.weights[c][self.COMMENT],
                    },
                    str(root_userID): {
                        "user_id": str(root_userID),
                        "timestamp": root_timestamp,
                        "Level": 0,
                        "retweedFrom": [{"user": str(root_userID), "prob": 1}],
                        "maxFrom": {"user": str(root_userID), "prob": 1},
                        "type": "root",
                    },
                }
            )
        return cascades

    # double version: link to the oldest or newest retweet
    def last_retweet(self, possibleChoices, times):
        id_last = np.argmax(times)
        lastFriend = possibleChoices[id_last]
        return lastFriend

    def first_retweet(self, possibleChoices, times):
        id_first = np.argmin(times)
        firstFriend = possibleChoices[id_first]
        return firstFriend

    def find_the_link(self, cascade, retwettatore, k, cascata_index, tweetsinfo):
        from_id = retwettatore
        possibleChoices = []
        status_error = []

        if self.data.get_exists(from_id)["crawled"] is True:
            for key in cascade.keys():
                risposta = self.data.get_is_friend(key, from_id)
                if risposta["status"] == 200:
                    if risposta["is_friend"] is True:
                        possibleChoices.append(key)
                elif risposta["status"] != 200:
                    status_error.append(key)

            if len(possibleChoices) != 0:

                times = [cascade[iden]["timestamp"] for iden in possibleChoices]
                if last is False:
                    friend = first_retweet(possibleChoices, times)
                else:
                    friend = last_retweet(possibleChoices, times)

                try:

                    cascade[retwettatore] = {
                        "user_id": retwettatore,
                        "timestamp": tweetsinfo["timestamp"].loc[k],
                        "retweedFrom": [
                            {"user": friend, "prob": (1 / len(possibleChoices))}
                        ],
                        "maxFrom": {"user": friend, "prob": (1 / len(possibleChoices))},
                        "type": "friends_list",
                    }
                except TypeError:

                    cascade[retwettatore] = {
                        "user_id": retwettatore,
                        "timestamp": tweetsinfo["timestamp"].loc[k],
                        "retweedFrom": [
                            {"user": friend, "prob": (1 / len(possibleChoices))}
                        ],
                        "maxFrom": {"user": friend, "prob": (1 / len(possibleChoices))},
                        "type": "friends_list",
                    }

                if len(status_error) != 0:
                    users_to_check_404.append(status_error)

            elif len(status_error) != 0 and len(possibleChoices) == 0:

                users_to_check_404.append(status_error)
            elif len(status_error) == 0 and len(possibleChoices) == 0:
                cascade[retwettatore] = {
                    "user_id": retwettatore,
                    "timestamp": tweetsinfo["timestamp"].loc[k],
                    "Level": "SP",
                    "retweedFrom": None,
                    "maxFrom": {"user": None, "prob": 1},
                    "probability": 1,
                    "type": "friends_list",
                }

        elif self.data.get_exists(int(from_id))["crawled"] is False:

            cascade["users_to_craw"].append(from_id)

        return cascade

    def habitual_friend(
        self, cascade, retwettatore, k, users_link, cascata_index, tweetsinfo
    ):
        location = tweetsinfo["timestamp"][tweetsinfo["user_id"] == retwettatore].index
        retweet_time = tweetsinfo["timestamp"].loc[location][
            tweetsinfo["user_id"] == retwettatore
        ]

        if retwettatore in users_link.keys():
            possible_link = [c for c in users_link[retwettatore].index]
            probabilities_link = [c for c in users_link[retwettatore].probabilities]
            dates = [
                tweetsinfo[tweetsinfo["user_id"] == c]["timestamp"]
                for c in possible_link
            ]

            prob_keep = []
            links_keep = []
            links = []
            max_p = {"user": "", "prob": 0}
            for d in range(0, len(dates)):

                if (
                    len(dates[d]) > 0
                    and retweet_time[retweet_time.index[0]]
                    > dates[d][dates[d].index[0]]
                    and probabilities_link[d] != 0
                ):
                    prob_keep.append(probabilities_link[d])
                    links_keep.append(possible_link[d])
                    links.append(
                        {"user": possible_link[d], "prob": probabilities_link[d]}
                    )
                    if probabilities_link[d] > max_p["prob"]:
                        max_p = {
                            "user": possible_link[d],
                            "prob": probabilities_link[d],
                        }

            if len(links_keep) != 0:
                cascade[retwettatore] = {
                    "user_id": retwettatore,
                    "timestamp": retweet_time[retweet_time.index[0]],
                    "retweedFrom": links,
                    "maxFrom": max_p,
                    "probability": pd.DataFrame(
                        {"user_id": links_keep, "prob": prob_keep}
                    ).to_json(orient="index"),
                    "type": "timeline",
                }

            elif len(links_keep) == 0:
                cascade = self.find_the_link(
                    cascade, retwettatore, k, cascata_index, tweetsinfo
                )
        elif retwettatore not in users_link.keys():
            cascade["users_to_craw"].append(retwettatore)
            cascade = self.find_the_link(
                cascade, retwettatore, k, cascata_index, tweetsinfo
            )
        return cascade

    def links_levels(self, cascades, tweetsinfo, all_users_link):
        for k in tqdm(range(0, len(list(tweetsinfo["user_id"])))):

            retwettatore = tweetsinfo["user_id"].iloc[k]
            for c in tqdm(range(0, len(all_users_link))):
                if retwettatore not in list(cascades[c].keys()):
                    cascades[c] = self.habitual_friend(
                        cascades[c], retwettatore, k, all_users_link[c], c, tweetsinfo
                    )

        return cascades

    def links_levels_parallelo(self, cascades, tweetsinfo, all_users_link):
        for k in tqdm(range(0, len(list(tweetsinfo["user_id"])))):
            retwettatore = tweetsinfo["user_id"].iloc[k]
            cascades = self.links_parallel_runs(
                enumerate(cascades), tweetsinfo, all_users_link, retwettatore, k
            )
        return cascades


# ----------------------------------------------------------------------------
# multiprocessing setup
cores = processes = mp.cpu_count()
cores = 1
# -------------------------------------------------------------------------------


def create_plot(cascade, filename):
    dot = Digraph(comment="cascade")
    for unlinked_user in cascade["users_to_craw"]:
        dot.node(str(unlinked_user))
    linked_users = list(cascade.keys())[2:]
    for linked_user in linked_users:
        dot.edge(str(cascade[linked_user]["maxFrom"]["user"]), str(linked_user))
    dot.render(os.path.join("output", str(filename)), view=False)


pool = None
if __name__ == "__main__":

    print("processors: ", cores)
    pool = mp.Pool(cores)

    # TODO: this should be changed to where you have the data, if you run it in a liacs DS machine you can use
    # data_root = os.path.join("data", "s1805819", "fakenewsnet_dataset")
    data_root = os.path.join("D:", "Onderzoek", "FakeNews", "fakenewsnet_dataset")
    user_timeline_root = os.path.join(data_root, "user_timeline_tweets")

    story_path = os.path.join(data_root, "politifact", "real")
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
                        try:
                            original_tweet_object = json.load(open(tweet_path, "rb"))
                            user_timeline_folder = os.path.join(user_timeline_root, story)
                            data_object = RetrieveData(
                                user_timeline_folder=user_timeline_folder,
                                original_tweet_object=original_tweet_object,
                                retweets_object=retweets_object,
                            )

                            # TODO: we should check what a good percentage is
                            if data_object.percentage_obtained > 85:
                                print(
                                    f"creating cascade wiht {data_object.percentage_obtained:.2f}% of the involved users crawled"
                                )
                                cascade = CreateCascade(data_object)
                                cascades = cascade.get_cascades()
                                create_plot(cascades, original_tweet_object["id"])
                        except Exception as e:
                            print(f'encountered a problem: {e}')
    pool.close()

import os
import json
import tqdm
from retrieveLostData import RetrieveData as collectOld


class RetrieveData:
    def __init__(self, user_timeline_folder, original_tweet_object, retweets_object, simple):
        self.user_timeline_folder = user_timeline_folder
        self.collectTimeline = collectOld()
        self.user_timelines = {}
        self.original_tweet_object = original_tweet_object
        self.retweets = retweets_object["retweets"]
        for retweet in tqdm.tqdm(self.retweets, desc="user profiles to collect"):
            if simple:
                user_timeline = {'replies': [], 'quotes': [], 'retweets':[]}
            else:
                user_timeline = self.read_user_timeline(
                    retweet["user"]["id"],
                    retweet["created_at"],
                    retweet["user"]["screen_name"],
                    self.collectTimeline
                )
            if user_timeline is not None:
                self.user_timelines[retweet["user"]["id"]] = user_timeline

        self.percentage_obtained = len(self.user_timelines)/len(retweets_object["retweets"]) * 100

    def read_user_timeline(self, user, created_at, screen_name, collectTimeline):
        user_timeline_filepath = os.path.join(self.user_timeline_folder, f"{user}.json")
        if not os.path.exists(self.user_timeline_folder):
            os.mkdir(self.user_timeline_folder)
        if not os.path.exists(user_timeline_filepath):
            # print(f"\ncollecting new data for {screen_name}")
            collectTimeline.gather_data(screen_name, user, created_at, user_timeline_filepath)
        else:
            # print(f"already collected data for {screen_name}")
            pass
        return json.load(open(user_timeline_filepath, "rb"))

    def get_tweet_object(self, user_id):
        if user_id == self.original_tweet_object["user"]["id"]:
            return self.original_tweet_object
        for r in self.retweets:
            if r["user"]["id"] == user_id:
                return r

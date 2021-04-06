import os
import json


class RetrieveData:
    def __init__(self, user_timeline_folder, original_tweet_object, retweets_object):
        self.user_timeline_folder = user_timeline_folder
        self.user_timelines = {}
        self.original_tweet_object = original_tweet_object
        self.retweets = retweets_object["retweets"]
        for retweet in self.retweets:
            user_timeline = self.read_user_timeline(retweet["user"]["id"])
            if user_timeline is not None:
                self.user_timelines[retweet["user"]["id"]] = user_timeline

        self.percentage_obtained = len(self.user_timelines)/len(retweets_object["retweets"]) * 100

    def read_user_timeline(self, user):
        user_timeline_filepath = os.path.join(self.user_timeline_folder, f"{user}.json")
        if os.path.exists(user_timeline_filepath):
            return json.load(open(user_timeline_filepath, "rb"))
        return None

    def get_retweet_matrix(self, users):
        # TODO: construct a matrix M, ixj; where |i| = len(users) and |j| = len(users)
        # TODO: M[i, j] = |retweets between user i and j|
        # TODO: ['matrix'][i]['autore'] is author of retweet (i.e. the person who retweeted it)
        # TODO: ['matrix'][i]['autore_originale'] is the original author of retweet (i.e. the person who created the tweet)
        # TODO: ['matrix']['docCount'] is the number of retweets

        authors = []
        original_authors = []
        doc_counts = []
        for user in users:
            if user in self.user_timelines.keys():
                user_timeline = self.user_timelines[user]
                retweet_dict = {}
                for tweet in user_timeline:
                    if "retweeted_status" in tweet.keys():
                        original_author = tweet["retweeted_status"]["user"]["id"]
                        if original_author not in retweet_dict:
                            retweet_dict[original_author] = 0
                        retweet_dict[original_author] += 1

                for original_author in retweet_dict.keys():
                    authors.append(user)
                    original_authors.append(original_author)
                    doc_counts.append(retweet_dict[original_author])
                # print(f'found {len(user_timeline)} timeline tweets for user {user}')
            # else:
            #     print(f'we have no timeline data for user {user}')
        assert len(authors) == len(original_authors) == len(doc_counts)
        return authors, original_authors, doc_counts


    def get_quote_matrix(self, users):
        # TODO: construct a matrix M, ixj; where |i| = len(users) and |j| = len(users)
        # TODO: M[i, j] = |quotes between user i and j|
        # TODO: ['matrix'][i]['autore'] is author of quote (i.e. the person who quoted it)
        # TODO: ['matrix'][i]['autore_originale'] is the original author of quote (i.e. the person who created the tweet)
        # TODO: ['matrix']['docCount'] is the number of quotes
        authors = []
        original_authors = []
        doc_counts = []
        for user in users:
            if user in self.user_timelines.keys():
                user_timeline = self.user_timelines[user]
                quoted_dict = {}
                for tweet in user_timeline:
                    if "retweeted_status" not in tweet.keys() and tweet["is_quote_status"] == True:
                        if "quoted_status" in tweet.keys():
                            original_author = tweet["quoted_status"]["user"]["id"]
                            if original_author not in quoted_dict:
                                quoted_dict[original_author] = 0
                            quoted_dict[original_author] += 1
                        # else:
                        #     print(f'original quoted tweet deleted, no way of knowing the original author')

                for original_author in quoted_dict.keys():
                    authors.append(user)
                    original_authors.append(original_author)
                    doc_counts.append(quoted_dict[original_author])
                # print(f'found {len(user_timeline)} timeline tweets for user {user}')
            # else:
            #     print(f'we have no timeline data for user {user}')
        assert len(authors) == len(original_authors) == len(doc_counts)
        return authors, original_authors, doc_counts

    def get_reply_matrix(self, users):
        # TODO: construct a matrix M, ixj; where |i| = len(users) and |j| = len(users)
        # TODO: M[i, j] = |replies between user i and j|
        # TODO: ['matrix'][i]['autore'] is author of reply (i.e. the person who replied to it)
        # TODO: ['matrix'][i]['autore_originale'] is the original author of tweet (i.e. the person who created the tweet)
        # TODO: ['matrix']['docCount'] is the number of replies
        authors = []
        original_authors = []
        doc_counts = []
        for user in users:
            if user in self.user_timelines.keys():
                user_timeline = self.user_timelines[user]
                reply_dict = {}
                for tweet in user_timeline:
                    if tweet["in_reply_to_user_id"] is not None:
                        original_author = tweet["in_reply_to_user_id"]
                        if original_author not in reply_dict:
                            reply_dict[original_author] = 0
                        reply_dict[original_author] += 1

                for original_author in reply_dict.keys():
                    authors.append(user)
                    original_authors.append(original_author)
                    doc_counts.append(reply_dict[original_author])
                # print(f'found {len(user_timeline)} timeline tweets for user {user}')
            # else:
            #     print(f'we have no timeline data for user {user}')
        assert len(authors) == len(original_authors) == len(doc_counts)
        return authors, original_authors, doc_counts

    def get_retweet_list(self):
        # TODO: should contain ['results']['users_id']
        # TODO: should contain ['results']['timestamp']
        return_object = {}
        results = []
        for rt in self.retweets:
            results.append({
                'user_id': rt['user']['id'],
                'date': rt['created_at']
            })
        return_object['results'] = results
        return return_object


    def get_tw_info(self, tweetID):

        # TODO: if tweet not found return ['status'] 404

        # TODO: root_tw_info['result']['content_date'] should contain date
        # TODO: root_tw_info['result']['content']['author']['id'] should contain id

        raise NotImplementedError

    def get_users_metrics(self, userID):
        # TODO: if correct ['status'] == 200
        # TODO: should contain ['friends_count']
        # TODO: should contain ['followers_count']
        # TODO: should contain ['favourites_count']
        # TODO: should contain ['statuses_count']
        return_object = {'status': 0}
        for rt in self.retweets:
            if rt['user']['id'] == userID:
                return_object['friends_count'] = rt['user']['friends_count']
                return_object['followers_count'] = rt['user']['followers_count']
                return_object['favourites_count'] = rt['user']['favourites_count']
                return_object['statuses_count'] = rt['user']['statuses_count']
                return_object['status'] = 200
                return return_object
        return return_object

    def get_is_friend(self, to_id, from_id):
        # TODO: return ['status'] == 200 if OK
        # TODO: return ['is_friend'] if user['to_id'] in list of friends of user['from_id']
        raise NotImplementedError

    def get_exists(self, from_id):
        # TODO: return ['crawled']True / False based on if the friends of the user are crawled
        return {'crawled': False}

    # TODO: seems to be not implemented in original work
    def get_retweets_users(self):
        raise NotImplementedError

    # TODO: seems to be not implemented in original work
    def get_quote_users(self):
        raise NotImplementedError

    # TODO: seems to be not implemented in original work
    def get_replies_users(self):
        raise NotImplementedError

import os
from searchtweets import ResultStream, load_credentials, gen_request_parameters
from datetime import datetime, timedelta
from TwitterAPI import TwitterAPI, OAuthType, TwitterRequestError, TwitterConnectionError
import requests
import time
import json



def create_query_obj(query_str: str, start_date: str, end_date: str):
    return gen_request_parameters(
        query=query_str,
        results_per_call=500,
        start_time=start_date,
        end_time=end_date,
        tweet_fields="created_at",
        expansions="author_id,referenced_tweets.id,referenced_tweets.id.author_id",
    )


def create_query_str(screen_name: str):
    return f"@{screen_name}"


def get_start_and_end_date(rt_date: str):
    try:
        end_time_obj = datetime.strptime(rt_date, '%a %b %d %H:%M:%S +0000 %Y')
        start_time_obj = (end_time_obj - timedelta(days=365))
        time_range = []
        for time in [start_time_obj, end_time_obj]:
            time_range.append(datetime.strftime(time, '%Y-%m-%d %H:%M'))
        return time_range
    except Exception as e:
        print(f"ERROR: {e}")


class RetrieveData:
    def __init__(self):
        self.KEY_FILE = "twitter_keys.yaml"
        self.academic_search_args = self._setup_twitter_api()

    def _setup_twitter_api(self):
        return load_credentials(os.path.join("resources", self.KEY_FILE), yaml_key="search_tweets_v2", env_overwrite=False)

    def gather_users(self, tweet_ids: [int]):
        author_ids = []
        if len(tweet_ids) == 0:
            return author_ids
        if len(tweet_ids) > 100:
            author_ids = self.gather_users(tweet_ids[100:])
            tweet_ids = tweet_ids[:100]

        try:
            api = TwitterAPI(
                consumer_key="--",
                consumer_secret="--",
                access_token_key="--",
                access_token_secret="--",
                auth_type=OAuthType.OAUTH1,
                api_version='2'
            )

            response = api.request(f'tweets', params={'ids': ','.join(tweet_ids), 'expansions': 'author_id'})

            for item in response:
                author_ids.append(item["author_id"])

            quota = response.get_quota()
            if quota['reset'] is not None:
                # oh shit, we have to wait
                now = datetime.today()
                reset_time = quota['reset']

                sleeptime_in_seconds = (reset_time-now).total_seconds()+5
                print(f'sleeping for {sleeptime_in_seconds} seconds')

                time.sleep(sleeptime_in_seconds)
            else:
                # print(f'{quota["remaining"]} API calls remaining')
                pass

        except TwitterRequestError as e:
            print(e.status_code)
            for msg in iter(e):
                print(msg)

        except TwitterConnectionError as e:
            print(e)

        except Exception as e:
            print(e)

        return author_ids


    def gather_data(self, screen_name: str,  user_id: int, rt_date: str, file_path: str):
        query_str = create_query_str(screen_name)
        # print(f'reconstructing timeline for @{screen_name}')

        time_range = get_start_and_end_date(rt_date)
        query_obj = create_query_obj(query_str, *time_range)
        rs = ResultStream(
            request_parameters=query_obj,
            # parameter changed from 2 -> 1 to avoid being ratelimited within the project timeline
            max_requests=1,
            **self.academic_search_args
        )
        inbound_timeline = []

        replies = []
        retweets = []
        quotes = []

        for tweet in rs.stream():
            if "author_id" not in tweet:
                if "tweets" in tweet:
                    # Tweets are found
                    for t in tweet["tweets"]:
                        if int(t["author_id"]) == user_id:
                            if "referenced_tweets" in t:
                                ref_tweets = t["referenced_tweets"]
                                for ref in ref_tweets:
                                    type = ref["type"]
                                    if type == "replied_to":
                                        replies.append(ref["id"])
                                    elif type == "quoted":
                                        quotes.append(ref["id"])
                            else:
                                # normal tweet, which holds no info on the information strength
                                pass
                        else:
                            if "referenced_tweets" not in t:
                                # the only way this situation can occur is when the tweet is retweeted by the autor
                                # and someone is replying to that retweet
                                retweets.append(t["author_id"])
                            else:
                                # this indicates a reply with a quote, or a reply of a reply
                                pass

        # print(f"done collecting the retweeted user objects, there are {len(retweets)} in total")

        # print(f"converting the {len(replies)} replied tweet objects to user ids")
        replies = self.gather_users(replies)
        # print(f"done collecting the replies user objects, there are {len(replies)} in total")

        # print(f"converting the {len(quotes)} quoted tweet objects to user ids")
        quotes = self.gather_users(quotes)
        # print(f"done collecting the quotes user objects, there are {len(quotes)} in total")

        # print(f"retweets: {len(retweets)}\treplies: {len(replies)}\tquotes: {len(quotes)}")

        dump_dict = {"replies": replies, "quotes": quotes, "retweets": retweets}
        json.dump(dump_dict, open(file_path, "w"))

if __name__ == '__main__':
    # no data found on user 236506960 -> brokenwing2005 who retweeted at Mon Jun 04 01:32:52 +0000 2018
    rt_date = "Mon Jun 04 01:32:52 +0000 2018"
    user_id = 236506960
    screen_name = "brokenwing2005"

    # rt_date = "Fri Apr 23 9:00:00 +0000 2021"
    # user_id = 367703310
    # screen_name = "thierrybaudet"

    data_handler = RetrieveData()
    data_handler.gather_data(screen_name, user_id, rt_date, f'output/{user_id}_just_checking.json')
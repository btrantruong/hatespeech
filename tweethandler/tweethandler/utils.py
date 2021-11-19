import logging 
import os 
import gzip 
import json
import pandas as pd 
import pickle 

def init_logging(currtime):
    log_filename = 'logfiles/logging_%s.log' %(currtime)
    logging.basicConfig(
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%m-%d %H:%M',
        filename = log_filename,
        level=logging.DEBUG
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #disable DEBUG level messages from urllib3
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def save_userdict(tweet_dict, out_path):
    with open(out_path, 'wb') as outf:
        try:
            pickle.dump(tweet_dict, outf, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Exception when saving tweets from dict, file %s' %out_path, e)
            pass

def dump_tweets(tweets_to_dump, out_path):
    with gzip.open(out_path, 'wb') as f:
        for tweet in tweets_to_dump:
            str_to_write = json.dumps(tweet) + '\n'
            f.write(str_to_write.encode('utf-8'))


def load_tweets(path):
    tweets = []
    with gzip.open(path) as f:
        for line in f:
            raw_tweet = json.loads(line)
            tweets.append(raw_tweet)
    return tweets

def get_logger(name):
    logging.basicConfig(
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt='%m-%d %H:%M',
        filename = name,
        level=logging.DEBUG
    )
    # Create handlers, add formatter to handlers, add handlers to the logger
    console = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s@%(name)s:%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.addHandler(console)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger

def run_scandir(dir, ext):    # dir: str, ext: list
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() == ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def feather_compliance(df):
    # make sure df is compliant with feather - feather doesnt serialize index
    df.reset_index(drop=True, inplace=True)
    if not df.index.equals(pd.RangeIndex.from_range(range(len(df)))):
        df.index = pd.RangeIndex.from_range(range(len(df)))
    return df

def save_df(data, outp): 
    df = pd.DataFrame.from_dict(data)
    df.to_parquet(outp)
    # df = feather_compliance(df_)
    # df.to_feather(outp)
    # logger.info('Finished saving df')
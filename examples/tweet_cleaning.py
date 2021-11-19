
ABSOLUTE_PATH='/Users/baott/hatespeech'
import os
import sys
sys.path.insert(1, os.path.join(ABSOLUTE_PATH, 'tweethandler'))
from tweethandler import lang_utils, extract_features
import tweethandler
import gzip
import json
import pandas as pd 

""" Option to save a corpus file (numbers, special char removed, all words from tweets separated by space)
    Take in .gz tweets, return a list of dict where each dict is a tweet object
    feature options for Moe random tweets:
    tid, uid, uscreen, uname, timestamp, rt, retweeted_tid, retweeted_uid, retweeted_uscreen, retweeted_uname, text
"""
def process_rawtwt(raw_inf, filter_en=True, verbose=False, get_hashtag=False,get_url=False,get_text=True):
    features = ['tid', 'uid', 'timestamp', 'rt', 'retweeted_tid', 'retweeted_uid','text']
    alltwts=[]
    total = 0
    missed = 0
    #construct uid - feature - count table with both native and non-native twts
    for line in raw_inf:
        try:
            raw_twt = json.loads(line)
            if 'limit' in raw_twt: continue #skips rate limit error objects
            #if 'retweeted_status' not in tweet: continue #skip non-retweets
            if isinstance(raw_twt,list):
                twt = raw_twt[0]
            twt = extract_features.keep_twtfeatures(raw_twt,HASHTAG=get_hashtag,URL=get_url, TEXT=get_text)
            cleaned_text = lang_utils.preprocess_text(twt['text'])
            twt['text'] = cleaned_text
            if verbose:
                print('-- Cleaned p: %s \n' %cleaned_text)
                print('-- En: %s \n' %lang_utils.is_english(cleaned_text))

            if filter_en & lang_utils.is_english(cleaned_text):
                final_twt_object = {feature: twt[feature] for feature in features if feature in twt.keys()}
                alltwts.append(final_twt_object)
            total +=1
        except Exception as e:
            #logger.error('Exception when parsing: %s, tweet: %s' %(e,twt))
            missed +=1
            pass
    #logger.info('Table length: %s/%s, (unable to parse: %s)' %(len(alltwts), total, missed))
    return alltwts

inpath ='examples/tweets2021-03-01.gz'
outpath= 'examples/cleaned_tweets2021-03-01.parquet'
with gzip.open(inpath, 'rb') as file:
    parsed_twttable = process_rawtwt(file,verbose=False)

twttable = pd.DataFrame.from_records(parsed_twttable)
twttable.to_parquet(outpath, engine='pyarrow')
print('Finish saving!')

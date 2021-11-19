""" Add filter criteria for tweet (e.g: is retweet or not, etc.) and return specified features 
    Options: 'hashtags','text','urls','timestamp'
    Urls are to be expanded later using df.apply to map URL expander module
    -- Covid data is preprocess(?), hence has different format, e.g: 'otherfields'
    """

HASHTAG = 'hashtags'
URL = 'urls'
TEXT = 'text'
TIMESTAMP = 'timestamp'
TWT_REGULAR =  True
#RETWEET = ['retweeted_tid','retweeted_uid','retweeted_uscreen','retweeted_uname']
def keep_twtfeatures(tweet, get_retweetinfo=True, timestamp=True, **kwargs):
    global TWT_REGULAR
    cleaned_tweet = {}

    if 'otherfields' in tweet:
        TWT_REGULAR = False 

    #always get tid and uid
    twt_user_info = get_user_tweet_idstr(tweet)
    for k,v in twt_user_info.items():
        cleaned_tweet[k] = v

    if timestamp is True:
        if TWT_REGULAR is False and 'timestamp_ms' in tweet['otherfields']:
            cleaned_tweet[TIMESTAMP] = tweet['otherfields']['timestamp_ms']
        elif 'timestamp_ms' in tweet: 
            cleaned_tweet[TIMESTAMP] = tweet['timestamp_ms']
    
    # check if a post is a tweet or retweet, then get entities from extended list or non-extended
    #retweet is the content inside tweet['retweeted_status']
    if 'retweeted_status' in tweet:
        cleaned_tweet['rt'] = 1 
        post = tweet['retweeted_status']
        if get_retweetinfo is True: 
            retwt_user_info = get_user_tweet_idstr(post, get_retweet=True)
            for k,v in retwt_user_info.items():
                cleaned_tweet[k] = v
    else: 
        cleaned_tweet['rt'] = 0
        post = tweet
        
    if kwargs['URL'] is True:
        cleaned_tweet[URL] = getfield_fromtweet(URL,post)
    
    if kwargs['HASHTAG'] is True:
        cleaned_tweet[HASHTAG] = getfield_fromtweet(HASHTAG,post)
    
    if kwargs['TEXT'] is True:
        cleaned_tweet[TEXT] = getfield_fromtweet(TEXT,post)

    return cleaned_tweet

def get_user_tweet_idstr(tweet, get_retweet=False):
    #get these fields = ['tid', 'uid', 'uscreen', 'uname']
    info_={}

    if 'id_str' in tweet:
        info_['tid'] = tweet['id_str']
    if 'user' in tweet:
        if 'id_str' in tweet['user']:
            info_['uid'] = tweet['user']['id_str']
        if 'screen_name' in tweet['user']:
            info_['uscreen'] = tweet['user']['screen_name']
        if 'name' in tweet['user']:
            info_['uname'] = tweet['user']['name']
    if get_retweet is True:
        info = {''.join(['retweeted_',k]): v for k,v in info_.items()}
    else:
        info = info_
    return info

def getfield_fromtweet(field, tweet):
    if 'extended_tweet' in tweet: 
        ext = tweet['extended_tweet']
        if field==HASHTAG and 'entities' in ext: 
            hashtags = [ht['text'] for ht in ext['entities']['hashtags']]
            return ','.join([str(elem) for elem in hashtags])
        if field==URL and 'entities' in ext: 
            all_urls = ext['entities']['urls']
            urls= [u['url'] for u in all_urls if "twitter.com" not in u['expanded_url']]
            return ','.join([str(elem) for elem in urls]) if len(urls)>0 else None
        if field==TEXT and 'full_text' in ext: 
            return ext['full_text']
    else:
        if field==HASHTAG and 'entities' in tweet: 
            hashtags = [ht['text'] for ht in tweet['entities']['hashtags']]
            return ','.join([str(elem) for elem in hashtags])
        if field==URL and 'entities' in tweet:
            all_urls = tweet['entities']['urls']
            urls= [u['expanded_url'] for u in all_urls if "twitter.com" not in u['expanded_url']]
            return ','.join([str(elem) for elem in urls]) if len(urls)>0 else None
        if field==TEXT and 'text' in tweet:
            return tweet['text']
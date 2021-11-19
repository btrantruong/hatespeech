import re
#python2: from urlparse import urlparse
from urllib.parse import urlparse
import preprocessor
import cld3

""" Wrapper function for preprocess package to clean text
    Make sure you've already installed the tweet-preprocess package: https://pypi.org/project/tweet-preprocessor/
"""
def preprocess_text(tweet_text):
    try:
        # preprocess removes emojis, url, mentions, numbers. Can be reconfigured to include those. 
        # Stopwords are not removed to retain readability when we want to manually check tweets.
        text = preprocessor.clean(tweet_text)
        special_char = '.@_!#$%^&*()<>?/\|}{~:;,[]"'
        tokens = text.split(' ')
        raw_text = " ".join([t for t in tokens if not has_numbers(t)])
        text = raw_text.lower().translate({ord(ch): ' ' for ch in '0123456789'+special_char})
        #removes spaces in-between words
        text = " ".join(text.split())
    except Exception as e:
        print('Exception in clean text!')
    return text

def remove_stopwords(text):
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    tokens = text.split(' ')
    clean_tokens = [t for t in tokens if t not in stopwords]
    cleaned_text = " ".join(clean_tokens)
    return cleaned_text

# cld3.get_language(string)
# LanguagePrediction(language='zh', probability=0.999969482421875, is_reliable=True, proportion=1.0)
def is_english(string):
    if string is None or string=='':
        return False
    lang, proba, isReliable, _ = cld3.get_language(string)
    english = lang == 'en'
    return isReliable and english

def has_numbers(string):
    return any(char.isdigit() for char in string)

""" ANOTHER ALTERNATIVE: Remove tweet entities such as mentions, hashtags, RT. Remove other properties such as number, stopwords. However doesn't guarantee to remove all emojis"""
def clean_tweet_text(tweet_text):
    try:
        special_char = '.@_!#$%^&*()<>?/\|}{~:;,[]"'
        stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
        #remove URLs, mentions, hashtags, words with numbers, emoji 
        tokens = tweet_text.split(' ')
        clean_tokens = [t for t in tokens if not is_url(t) and is_natural_lang(t) and not has_numbers(t) and not is_emoji(t) and t not in stopwords]
        raw_text = " ".join(clean_tokens)
        text = raw_text.lower().translate({ord(ch): ' ' for ch in '0123456789'+special_char})
        #strips out line ends
        text = text.rstrip()
        #text = re.sub('[()!?]', ' ', text) #strip punctuations
        #cleaned_text = re.sub('\[.*?\]',' ',text)
    except Exception as e:
        print('Exception in clean text!')
    return text


def is_emoji(string):
    emoji = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    if emoji.match(string) is not None:
        return True
    else:
        return False
#Return True if token doesn't contain hashtags, mentions, possible emojis 
def is_natural_lang(string):
    if '@' not in string and '#' not in string :
        # and '\u' not in string
        return True
    else:
        return False
def is_url(string):
    return bool(urlparse(string).scheme) and bool(urlparse(string).path)


ABSOLUTE_PATH='/Users/baott/hatespeech'
import os
import sys
sys.path.insert(1, os.path.join(ABSOLUTE_PATH, 'tweethandler'))
from tweethandler import lang_utils, extract_features

example_tweet = '@sinowicked Giving a shot here, i bake cakes with edible flowers, that sometimes does looks like a bouquet of flowers, the best part is, u can eat it 🌸 https://t.co/EoSfswd2wv https://t.co/70BCb3SpmJ'
cleaned_tweet = lang_utils.preprocess_text(example_tweet)
print('---BEFORE: \n %s \n ---AFTER: \n %s' %(example_tweet,cleaned_tweet))
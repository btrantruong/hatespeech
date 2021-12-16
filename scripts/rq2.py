""" Code for the first part of RQ2. Calculates 3 types of sentiment score (Nltk, Textblob and Flair).
    Train and test a n-gram Logistic regression model (analyzer is char or word) with these scores. 
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import os
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import operator 
import numpy as np
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence 
import pandas as pd
import scipy.sparse
model_path = "exps/trained_models"

nltk.download('vader_lexicon')
FLAIR_SIA = TextClassifier.load('en-sentiment')

def read_dataset(x_path, y_path):
    # returns an array of strings and an array of labels
    f_x = open(x_path,'r')
    X = [x.strip() for x in f_x.readlines()]

    f_y = open(y_path, 'r')
    Y = [y.strip() for y in f_y.readlines()]
    return X, Y

def map_sentiment_nltk(df):
  sia = SentimentIntensityAnalyzer()
  df['nltk_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])
  df['nltk_sentiment'] = np.select([df['nltk_score']<0, df['nltk_score']==0, df['nltk_score']>0], ['neg','neu','pos'])
  return df

def map_sentiment_textblob(df):
  df['textblob_score'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
  df['textblob_sentiment'] = np.select([df['textblob_score']<0, df['textblob_score']==0, df['textblob_score']>0], ['neg','neu','pos'])
  return df

def flair_prediction(x):
  sentence=Sentence(x)
  FLAIR_SIA.predict(sentence)
  score = sentence.labels
  #convert flair.Label object to str
  return str(score[0])

def split_flair_predsentiment(x):
  #example: [NEGATIVE (0.9999)]
  sen = x.split(' ')[0]
  if 'NEGATIVE' in sen:
    return "neg"
  elif 'POSITIVE' in sen:
    return "pos"
  else:
    return "neg"

def split_flair_predscore(x):
  #example: 'NEGATIVE (0.9999)'
  score_str = x.split(' ')[1]
  return float(score_str.strip("()"))

def map_sentiment_flair(df):
  df['flair'] = df['text'].apply(flair_prediction)
  df['flair_sentiment'] = df['flair'].apply(split_flair_predsentiment)
  df['flair_score'] = df['flair'].apply(split_flair_predscore)
  return df


def make_train_testdf(X, Y,sentiment='all'):
  df_text = pd.DataFrame(columns=['text', 'label'])
  df_text['text'] = X
  df_text['label'] = Y
  df_text = df_text.astype({'label':int})
  df_text['label'] = df_text['label'].apply(lambda x: 0 if x==1 else 1)
  df_text = map_sentiment_nltk(df_text)
  df_text = map_sentiment_textblob(df_text)
  if sentiment=='all':
    df_text = map_sentiment_flair(df_text)
  df_text.head()
  return df_text

#'nltk_score'
def run_model(model,analyzer, train_df, test_df, sentiment_score_col='nltk_score'):
    print('--Run model with analyzer: %s and sentiment: %s...' %(analyzer, sentiment_score_col))
    vectorizer = CountVectorizer(analyzer = analyzer, ngram_range = (1,4)) # / 5000 / 10000)
    
    train_X = scipy.sparse.hstack((vectorizer.fit_transform(train_df['text'].values),train_df[[sentiment_score_col]].values),format='csr')
    train_Y = train_df['label'].values

    test_X = scipy.sparse.hstack((vectorizer.transform(test_df['text'].values),test_df[[sentiment_score_col]].values),format='csr')
    test_Y = test_df['label'].values
    print('Finish building data, now train...')
    # model train and prediction
    model.fit(train_X, train_Y)
    print('Fit model...')
    preds = model.predict(test_X)
    print(classification_report(test_Y, preds, digits=6))
    joblib.dump(model, os.path.join(model_path, "logistic_%s_%s.pkl" %(analyzer, sentiment_score_col)))

exp_path = "data/waseem/"
train_filepath = os.path.join(exp_path, "waseemtrain_cleaned_no_empty.txt")
Ytrain_filepath = os.path.join(exp_path, "waseemtrainGold_cleaned_no_empty.txt")
test_filepath = os.path.join(exp_path, "waseemtest_cleaned_no_empty.txt")
Ytest_filepath = os.path.join(exp_path, "waseemtestGold_cleaned_no_empty.txt")
x_train, y_train = read_dataset(train_filepath, Ytrain_filepath)
x_test, y_test = read_dataset(test_filepath, Ytest_filepath)
train_df = make_train_testdf(x_train, y_train,sentiment='all')
test_df = make_train_testdf(x_test, y_test,sentiment='all')

sentiments = ['nltk_score', 'flair_score', 'textblob_score']
analyzers= ['word', 'char']

for analyzer in analyzers:
  for sentiment_type in sentiments:
    model = LogisticRegression()
    run_model(model,analyzer, train_df, test_df, sentiment_score_col=sentiment_type)


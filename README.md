# hatespeech
This repo is the accompaniment of a final project for the class L645: Advance Natural Language Processing. 

# Overview
The code are organized according to the research question (RQ) it is written to answer. e.g., `rq1.py` and files prepended by `rq1` answers RQ1. For more details, refer to the written final report.
  - RQ1: Can we replicate results from the original paper by just using word and char n-gram features for hate-speech classification?
  - RQ2: Is sentiment useful for n-gram-based hate-speech classification?
  - RQ3: Would context-dependent embedding features improve hate-speech detection?
  - RQ4: How does the best model from RQ1, RQ2 and RQ3 generalize on new annotated data? 

# Data 
`data/waseem` contains cleaned text data from the original paper: <a href='https://github.com/zeeraktalat/hatespeech'> Github link </a>

`data/annotated` contains our annotated data. This dataset consists of tweets collected using the Twitter API from 07-01-2021 to 11-29-2021 using a list of hashtags adopted from Waseem & Hovy's paper. 

## Setup & requirements 
For the tweet cleaning code to run properly, make sure you've already installed the tweet-preprocess package: https://pypi.org/project/tweet-preprocessor

For `rq2.py` code to run properly, please install the following packages: textblob, nltk and flair. 

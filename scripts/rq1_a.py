from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import os
import joblib


# change it to read waseem data
def read_dataset(x_path, y_path):
    """
    File_path should be a string that represents the filepath
    where the movie dataset can be found

    This returns an array of strings and an array of labels
    """
    f_x = open(x_path,'r')
    X = [x.strip() for x in f_x.readlines()]

    f_y = open(y_path, 'r')
    Y = [y.strip() for y in f_y.readlines()]
    return X, Y

# create a logistic regression classifier
model = LogisticRegression()

# generating part of speech features
print("Reading in dataset...")
waseem_dir = 'waseem'
train_text_data, train_Y = read_dataset(os.path.join(waseem_dir,'waseemtrain_cleaned_no_empty.txt'),os.path.join(waseem_dir,'waseemtrainGold_cleaned_no_empty.txt') )
test_text_data, test_Y = read_dataset(os.path.join('annotations','FINAL_X.txt'),os.path.join('annotations','FINAL_Y.txt') )
print(Counter(train_Y))
  #Now we need to extract features from the text data
print("Extracting features...")
  # To run a bag-of-words feature extractor in sklearn we first initialize the vectorizer
  # Check out the documentation to see all the different settings that can be used
  # In this case, we are doing analysis at the word level with unigrams, bigrams and trigrams
#vectorizer = CountVectorizer(analyzer='word',ngram_range=(1,3)) #TfidfVertorizer(analyzer = 'word', ngram_range(1,3), max_features = 1000 / 5000 / 10000)

for analyzer in ['word', 'char']:
    vectorizer = CountVectorizer(analyzer = analyzer, ngram_range = (1,4)) # / 5000 / 10000)
      # Now we run fit transform on the vectorizer
      # this will figure out what the unique n-grams are and return the feature matrix given these
    #train_X = vectorizer.fit_transform(train_text_data)
    train_X = vectorizer.fit_transform(train_text_data)
      # when we run over the test data though, we don't want to find new unique n-grams
      # This time, instead of running fit_transform, we simply run transform
    #test_X = vectorizer.transform(test_text_data)
    test_X = vectorizer.transform(test_text_data)

    # model train and prediction
    model.fit(train_X, train_Y)
    #Now we have a full model! Let's predict on the test dataset
    preds = model.predict(test_X)
    #Let's print some results to see how we did!
    print(classification_report(test_Y, preds, digits=6))
    joblib.dump(model, "logisitic_{}.pkl".format(analyzer))


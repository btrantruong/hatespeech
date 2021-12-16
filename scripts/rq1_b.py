#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import joblib
h = 0.02  # step size in the mesh
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import os

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
   # "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]


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

for analyzer in ['char']: # 'char_wb']: # word
    vectorizer = CountVectorizer(analyzer = analyzer, ngram_range = (1,4)) # / 5000 / 10000)
      # Now we run fit transform on the vectorizer
      # this will figure out what the unique n-grams are and return the feature matrix given these
    #train_X = vectorizer.fit_transform(train_text_data)
    train_X = vectorizer.fit_transform(train_text_data)
      # when we run over the test data though, we don't want to find new unique n-grams
      # This time, instead of running fit_transform, we simply run transform
    #test_X = vectorizer.transform(test_text_data)
    test_X = vectorizer.transform(test_text_data)


    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        try:
            clf.fit(train_X, train_Y)
            score = clf.score(test_X, test_Y)
            preds = clf.predict(test_X)
            print(name)
            print(classification_report(test_Y, preds, digits=6))
            joblib.dump(clf, "{}_{}.pkl".format(analyzer, name))
        except:
            print(name, 'have error')



        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].



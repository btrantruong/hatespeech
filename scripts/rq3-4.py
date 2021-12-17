import ast
import preprocessor as p

import torch
from transformers import BertTokenizerFast, BertModel
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
import pandas as pd
import re

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import joblib


def clean_tweets(df, field='tweets'):
  df[field] = df[field].apply(lambda x: x.lower())
  df[field] = df[field].apply(lambda x: x.replace('\d+', ''))
  df[field] = df[field].apply(lambda x: re.sub(r'[^\w\s]', '', (x)))
  df[field] = df[field].apply(lambda x: p.clean(x))


def read_data(rows, tweet_field, label_field):
    data = {}
    for row in rows:
        with open(row[0]) as f:
            lines = [x.strip() for x in f.readlines()]
            data[row[2]] = pd.DataFrame(data=[], columns=[tweet_field, label_field])
            data[row[2]][tweet_field] = lines
            f.close()
        with open(row[1]) as f:
            lines = [x.strip() for x in f.readlines()]
            data[row[2]][label_field] = lines
            f.close()

    return data

def get_sentence_embeddings(rows, tweet_field, label_field, path_data):
    list_sentences = [list(rows['test'][tweet_field]),
                      list(rows['train'][tweet_field]),
                      list(rows['annotated'][tweet_field])
                      ]
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
    model.eval()
    data_dict = {}

    batch_size = 100
    i = 0
    names = ['test', 'train', 'annotated']
    labels = [list(rows['test'][label_field]),
              list(rows['train'][label_field]),
              list(rows['annotated'][label_field])
              ]

    for sentences in list_sentences:
        all_data = []
        for idx in range(0, len(sentences), batch_size):
            batch = sentences[idx: min(len(sentences), idx + batch_size)]

            encoded = tokenizer.batch_encode_plus(batch,
                                                  max_length=50,
                                                  padding='max_length',
                                                  truncation=True)

            encoded = {key: torch.LongTensor(value) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = model(**encoded)
            lhs = outputs[2]  # all embedding in 13 layer

            # layers, # batches, # tokens, # features]
            token_embeddings = torch.stack(lhs, dim=0)
            token_embeddings = token_embeddings.permute(1, 0, 2, 3)

            for embedding in token_embeddings:
                embedding = embedding[-4:]
                sentence_embedding = torch.sum(embedding, dim=0)
                sentence_embedding = torch.mean(sentence_embedding, dim=0)

                all_data.append(sentence_embedding.tolist())

        data_dict[names[i]] = pd.DataFrame(columns=['embeddings', 'sentences', 'label'])
        data_dict[names[i]]['embeddings'] = all_data
        data_dict[names[i]]['sentences'] = sentences
        data_dict[names[i]]['label'] = labels[i]

        data_dict[names[i]].to_csv(f'{path_data}{names[i]}_embeddings.csv', index=False)
        i = i + 1

def classification_report_csv(report, text, name, path_result):
  report_data = []

  for key in report:
    row ={}
    if key == 'accuracy':
      row['class'] = key
      row['precision'] = ''
      row['recall'] = ''
      row['f1_score'] = ''
      row['support'] = ''
      row['classifier'] = name
      row['accuracy'] = report[key]
    else:
      row['class'] = key
      row['precision'] = report[key]['precision']
      row['recall'] = report[key]['recall']
      row['f1_score'] = report[key]['f1-score']
      row['support'] = report[key]['support']
      row['classifier'] = name
      row['accuracy'] = ''

    report_data.append(row)

  dataframe = pd.DataFrame.from_dict(report_data)
  dataframe.to_csv(f'{path_result}classification_report_{text}_{name}.csv', index = False)

def classify(train, test, annotated, embedding_field, label_field, path_result):
    names = [
        "Logistic Regression",
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Decision Tree",
        "Random Forest",
        "AdaBoost",
        "Naive Bayes",
    ]

    penalty = 'l2'
    C = 1.0
    class_weight = 'balanced'
    random_state = 2018
    solver = 'liblinear'
    classifiers = [
        LogisticRegression(penalty=penalty,
                           C=C,
                           class_weight=class_weight,
                           random_state=random_state,
                           solver=solver,
                           ),

        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
    ]

    X_train = list(train[embedding_field])
    X_test = list(test[embedding_field])
    y_test = list(test[label_field])
    y_train = list(train[label_field])
    X_annotated = list(annotated[embedding_field])
    y_annotated = list(annotated[label_field])


    for name, clf in zip(names, classifiers):
        try:
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            print(score)

            preds = clf.predict(X_test)
            report = classification_report(y_test, preds, digits=6, output_dict=True)
            name = name.replace(' ', '-')

            classification_report_csv(report, 'test', name, path_result)

            annot_preds = clf.predict(X_annotated)

            report = classification_report(y_annotated, annot_preds, digits=6, output_dict=True)

            classification_report_csv(report, 'annotated', name, path_result)

            joblib.dump(clf, f'{path_result}{name}.pkl')
        except Exception as e:
            print(e)


def main():
    path_data = './../data/'
    path_result = './../results/'
    test_path = f'{path_data}waseem/test.txt'
    test_path_label = f'{path_data}waseem/testGold.txt'
    train_path = f'{path_data}waseem/wtrain.txt'
    train_path_label = f'{path_data}waseem/rainGold.txt'
    annotated_path = f'{path_data}annotated.txt'
    annotated_path_label =  f'{path_data}waseem/labels.txt'

    rows = [[test_path, test_path_label, 'test'],
            [train_path, train_path_label, 'train'],
            [annotated_path, annotated_path_label, 'annotated']
            ]
    data = read_data(rows)
    df_train = data['train']
    df_test = data['test']

    df_train.loc[df_train['label'] == '2', 'label'] = '0'
    df_test.loc[df_test['label'] == '2', 'label'] = '0'

    clean_tweets(df_test)
    clean_tweets(df_train)
    clean_tweets(data['annotated'])

    get_sentence_embeddings(data, 'tweets', 'label')

    df_train_embeddings = pd.read_csv(f'{path_data}train_embeddings.csv')
    df_test_embeddings = pd.read_csv(f'{path_data}test_embeddings.csv')
    df_annotated_embeddings = pd.read_csv(f'{path_data}annotated_embeddings.csv')

    df_test_embeddings['embeddings'] = df_test_embeddings['embeddings'].apply(
        lambda x: ast.literal_eval(x))
    df_train_embeddings['embeddings'] = df_train_embeddings['embeddings'].apply(
        lambda x: ast.literal_eval(x))
    df_annotated_embeddings['embeddings'] = df_annotated_embeddings['embeddings'].apply(
        lambda x: ast.literal_eval(x))

    classify(df_train_embeddings,
             df_test_embeddings,
             df_annotated_embeddings,
             'embeddings',
             'label',
             path_result
             )


main()

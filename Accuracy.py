import pandas as pd
from os import path
import json
import csv
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.pipeline import Pipeline
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from sklearn.model_selection import KFold
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

global df

# Read training dataset
df = pd.read_csv('corpus.csv')

col = ['label', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]
X_texts = list(df['text'])
Y_labels = list(df['label'])
X, y = [], []
for sentence, label in zip(df['text'], df['label']):
    X.append(sentence.split(' '))
    y.append(label)
X, y = np.array(X), np.array(y)

# train the model using all the sentences == X
model = None
if path.isfile('word2vec.model'):
    model = Word2Vec.load("word2vec.model")
else:
    model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
    model.save("word2vec.model")
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)}


# vectorizor classes
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(next(iter(word2vec.values())))
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

    # ---------------------------
    # <<<<<<<------remove------>>>>>
    # ---------------------------- 


# tf-idf version
# class TfidfEmbeddingVectorizer(object):
#     def __init__(self, word2vec):
#         self.word2vec = word2vec
#         self.word2weight = None
#         if len(word2vec)>0:
#             self.dim=len(next(iter(word2vec.values())))
#         else:
#             self.dim=0

#     def fit(self, X, y):
#         tfidf = TfidfVectorizer(analyzer=lambda x: x)
#         tfidf.fit(X)
#         # if a word was never seen - so the default idf is the max of 
#         # known idf's
#         max_idf = max(tfidf.idf_)
#         self.word2weight = defaultdict(
#             lambda: max_idf, 
#             [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

#         return self

#     def transform(self, X):
#         return np.array([
#                 np.mean([self.word2vec[w] * self.word2weight[w]
#                          for w in words if w in self.word2vec] or
#                         [np.zeros(self.dim)], axis=0)
#                 for words in X
#             ])


# -------------------------------------------------------
# model  definetions
bern_nb = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), ("bernoulli nb", BernoulliNB())])
svc_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), ("linear svc", SVC(kernel="linear"))])
LR_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                   ("extra trees", LogisticRegression(random_state=0, max_iter=5000))])
etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])

# model list
lst_models = [
    ("logR_w2v ", LR_w2v),
    ("Nb_w2v", bern_nb),
    ("svc_w2v ", svc_w2v),
    ("ensemble ", etree_w2v)
]

scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in lst_models]
scores_sorted = sorted(scores, key=lambda x: -x[1])
print("\n \tCross Val Score \n")
print(tabulate(scores_sorted, floatfmt=".4f", headers=("Model Names", 'scores')))


def GetAccuracyScore(model_list, texts, labels):
    tbl_acc_score = []
    #print("\n \tAccuracy Score \n")
    for name, model in model_list:
        acc_scores = []
        # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.8, random_state=0)
        kfold = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kfold.split(texts, labels):
            X_train, X_test = texts[train_index], texts[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            acc_scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
        #print("#- ", name, "\t", np.mean(acc_scores))
        tbl_acc_score.append({'model': model, 'model_name': name, 'accuracy': np.mean(acc_scores)})
    return tbl_acc_score


accuracy_score = GetAccuracyScore(lst_models, X, y)


def GetSelectedModel(accuracy_scores):
    selectedScore = None
    selectedModel = None
    selectedModelName = None

    dictAcc_score = accuracy_scores
    for line in dictAcc_score:
        if selectedScore == None:
            selectedScore = line["accuracy"]
            selectedModel = line["model"]
            selectedModelName = line["model_name"]
        elif line["accuracy"] > selectedScore:
            selectedScore = line["accuracy"]
            selectedModel = line["model"]
            selectedModelName = line["model_name"]
    return selectedModel, selectedScore, selectedModelName


selected_model, selected_score, selected_model_name = GetSelectedModel(accuracy_score)
print("\n selected_model --- ", selected_model_name, "\n")


def writeJson(selectedmodel, _texts, _labels):
    model = selectedmodel
    model.fit(_texts, _labels)
    categorized = {}
    with open("@AmilDilshan_1X.csv", 'r') as data_file:
        reader = csv.reader(data_file, delimiter='\t')
        count = 0
        for line in reader:
            for field in line:
                print(field, ":")
                cat = model.predict([field])
                print(cat)
                if cat[0] not in categorized.keys():
                    categorized[cat[0]] = [field]
                else:
                    categorized[cat[0]].append(field)
                count = count + 1
    with open('data2223.json', 'w') as f:
        json.dump(categorized, f, ensure_ascii=False)
    print("+++++++++++++++++++++++++++++++")
    print("number of rows... ", count)
    print("+++++++++++++++++++++++++++++++")


writeJson(selected_model, X_texts, Y_labels)

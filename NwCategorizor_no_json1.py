import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from collections import defaultdict

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from tabulate import tabulate

# models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier

global df

# Read the training dataset
df = pd.read_csv('corpus.csv')

col = ['label', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]
txtArrList = list(df['text'])
X, y = [], []
print("----sentences----")
for sentence, label in zip(df['text'], df['label']):
    X.append(sentence.split(' '))
    y.append(label)
X, y = np.array(X), np.array(y)  # still optional phase 001


# train the model using all the sentences == X
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.vectors)}


# print(w2v)


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


# tf-idf version
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(next(iter(word2vec.values())))
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# these two uses the pre defined classes (word2vec classes)

etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                            ("extra trees",    ExtraTreesClassifier(n_estimators=200))])
nb_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("nb", MultinomialNB())])
svm_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("svm", LinearSVC())])
lr_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("lr", LogisticRegression(verbose=1,
                       solver='liblinear',
                       random_state=0,
                       C=5,
                       penalty='l2',
                       max_iter=1000))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                      ("extra trees", ExtraTreesClassifier(n_estimators=200))])

# model list
lst_models = [


    ("nb_w2v", nb_w2v),
    ("svm_w2v", svm_w2v),
    ("lr_w2v", lr_w2v),
    ("w2v", etree_w2v),

]

scores = [(name, cross_val_score(model, X, y, cv=5).mean()) for name, model in lst_models]
scores_sorted = sorted(scores, key=lambda x: -x[1])

print(tabulate(scores_sorted, floatfmt=".4f", headers=("Model Names", 'scores')))


def GetAccuracyScore(model_list, texts, labels):  # texts == X and labels == y
    tbl_acc_score = []

    for name, model in model_list:
        acc_scores = []
        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=None)
        for train_index, test_index in sss.split(texts, labels):
            X_train, X_test = texts[train_index], texts[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            acc_scores.append(accuracy_score(model.fit(X_train, y_train).predict(X_test), y_test))
            classification=classification_report(model.fit(X_train, y_train).predict(X_test), y_test)
        tbl_acc_score.append({'model': name, 'accuracy': np.mean(acc_scores)})
        print(classification)
    print(tbl_acc_score)
    return tbl_acc_score


GetAccuracyScore(lst_models, X, y)

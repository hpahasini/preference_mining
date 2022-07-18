from collections import defaultdict

import numpy as np
import gensim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

global df

# Read the training dataset
df = pd.read_csv('corpus_sm.csv')

col = ['label', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]
X = list(df['text'])
Y = list(df['label'])
model = gensim.models.Word2Vec(X, size = 100)
w2v = dict(zip(model.wv.index2word, model.wv.vectors))

print("text ----",X)
print("label ----",Y)
print(w2v)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
#         self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

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

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

etree_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_w2v_tfidf = Pipeline([
    ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
    ("extra trees", ExtraTreesClassifier(n_estimators=200))])

print(etree_w2v)
print(etree_w2v_tfidf)
etree_w2v.fit(X,Y)
test_X = ['home wave beachlifemindset oceanview livingforthenextbeachday peacelovebeach ice ocean crystal body of water solid ',
          'mind in always beach beachlifemindset thinkbeachythoughts photo outerbanks book product information creation text','food is awsome','that dog is cruel','fried rice','kottu','mango trees are beautiful']
prediction = etree_w2v.predict(test_X)
print(prediction)






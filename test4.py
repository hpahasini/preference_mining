from gensim.models import Word2Vec
import pandas as pd
import numpy as np
# example of a super learner model for regression
from math import sqrt
from numpy import hstack
from numpy import vstack
from numpy import asarray
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


# create a list of base-models
def get_models():
    models = list()
    models.append(LinearRegression())

    return models


# collect out of fold predictions form k-fold cross validation
def get_out_of_fold_predictions(X, y, models):
    meta_X, meta_y = list(), list()
    # define split of data
    kfold = KFold(n_splits=10, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        fold_yhats = list()
        # get data
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        meta_y.extend(test_y)
        # fit and make predictions with each sub-model
        for model in models:
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            # store columns
            fold_yhats.append(yhat.reshape(len(yhat), 1))
        # store fold yhats as columns
        meta_X.append(hstack(fold_yhats))
    return vstack(meta_X), asarray(meta_y)


# fit all base models on the training dataset
def fit_base_models(X, y, models):
    for model in models:
        model.fit(X, y)


# fit a meta model
def fit_meta_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# evaluate a list of models on a dataset
def evaluate_models(X, y, models):
    for model in models:
        yhat = model.predict(X)
        mse = mean_squared_error(y, yhat)
        print('%s: RMSE %.3f' % (model.__class__.__name__, sqrt(mse)))


# make predictions with stacked model
def super_learner_predictions(X, models, meta_model):
    meta_X = list()
    for model in models:
        yhat = model.predict(X)
        meta_X.append(yhat.reshape(len(yhat), 1))
    meta_X = hstack(meta_X)
    # predict
    return meta_model.predict(meta_X)


global df
# Read the training dataset
df = pd.read_csv('corpus_sm.csv')
col = ['label', 'text']
df = df[col]
texts = list(df[pd.notnull(df['text'])])
labels = list(df[pd.notnull(df['label'])])
print("texts \n", texts)
print("\n labels \n", labels)


def sent_vectorizer(sent_list, model):
    sentences = sent_list
    X = []
    for sentence in sentences:
        sent_vec = []
        numw = 0
        for w in sentence:
            try:
                if numw == 0:
                    sent_vec = model[w]
                else:
                    sent_vec = np.add(sent_vec, model[w])
                numw += 1
            except:
                pass
        # to model fit it requires positive
        X.append(abs(np.asarray(sent_vec) / numw))
    return X


model = Word2Vec(texts, min_count=1)
X = sent_vectorizer(texts, model)
y = sent_vectorizer(labels, model)

print("\n texts vectored \n", X)
print("\n labels vectored \n", y)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.50)
# print('Train', X.shape, y.shape, 'Test', X_val.shape, y_val.shape)
models = get_models()

meta_X, meta_y = get_out_of_fold_predictions(X, y, models)
print(meta_X)
# print('Meta ', meta_X.shape, meta_y.shape)
# fit base model
fit_base_models(X, y, models)

# fit the meta model
meta_model = fit_meta_model(meta_X, meta_y)
# evaluate base models
evaluate_models(X_val, y_val, models)
# evaluate meta model
yhat = super_learner_predictions(X_val, models, meta_model)
print('Super Learner: RMSE %.3f' % (sqrt(mean_squared_error(y_val, yhat))))

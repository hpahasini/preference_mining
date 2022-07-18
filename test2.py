import csv
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

global df

# Read the training dataset
df = pd.read_csv('corpus.csv')

col = ['label', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True,
                        min_df=5,
                        norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 2),
                        stop_words='english')
features = tfidf.fit_transform(df.text).toarray()

Train_X, Test_X, Train_Y, Test_Y = train_test_split(df['text'],
                                                    df['label'],
                                                    test_size=0.2,random_state=None)
Train_X, Val_X, Train_Y, Val_Y = train_test_split(Train_X,
                                                    Train_Y,
                                                    test_size=0.2,random_state=None)

Encoder = LabelEncoder()
Train_Y_En = Encoder.fit_transform(Train_Y)
Test_Y_En = Encoder.fit_transform(Test_Y)

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df['text'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(Train_X)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


model_predict1 = LinearSVC().fit(Train_X_Tfidf, Train_Y_En)
predictions1 = model_predict1.predict

model_predict2 = MultinomialNB().fit(Train_X_Tfidf, Train_Y_En)
predictions2 = model_predict2.predict

ensemble = LinearRegression().fit(predictions1, predictions2)

predictions3 = model_predict1.predict(Test_X_Tfidf)
predictions4 = model_predict2.predict(Test_X_Tfidf)

ensemble = LogisticRegression().predict(predictions3,predictions4)
accuracyScore = accuracy_score(ensemble, Test_Y_En)

print(accuracyScore)



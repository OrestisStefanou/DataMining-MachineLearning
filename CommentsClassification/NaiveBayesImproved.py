# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:56:30 2020

@author: orestis
"""
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

DataLocation = './data.csv'

df = pd.read_csv(DataLocation,header=None,names=['Comment','Class'])

X_train,X_test,y_train,y_test = train_test_split(df['Comment'],df['Class'], random_state=1)

cv = CountVectorizer(strip_accents='ascii',token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',lowercase=True,stop_words='english')

X_train_cv=cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)

print('Accuracy score: ',accuracy_score(y_test, predictions))
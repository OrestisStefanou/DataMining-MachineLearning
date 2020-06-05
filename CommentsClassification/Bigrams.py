# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:34:22 2020

@author: orestis
"""
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import nltk
import re

lemmatizer = WordNetLemmatizer() 
DataLocation = './data.csv'

first_df = pd.read_csv(DataLocation,header=None,names=['Comment','Class'])

comments_biagrams=list()
#Create a new list with the bigrams of the comments
for i in range(len(first_df['Comment'])):
    txt1 = first_df['Comment'][i]
    # initializing list  
    test_list = [txt1] 
      
    # using list comprehension + enumerate() + split() 
    # for Bigram formation 
    res = [(x, i.split()[j + 1]) for i in test_list  
           for j, x in enumerate(i.split()) if j < len(i.split()) - 1] 
    
    new_comment = ''
    for x in res:
        #Create a new comment with the lemmatized biagrams
        new_comment = new_comment + lemmatizer.lemmatize(x[0]) + lemmatizer.lemmatize(x[1]) + ' '
    comments_biagrams.append(new_comment)

#Create a new dataset with the biagrams
class_values = first_df['Class']
DataSet = list(zip(comments_biagrams,class_values))
df = pd.DataFrame(data=DataSet,columns=['Comment','Class'])

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), stop_words='english')

features = tfidf.fit_transform(df.Comment).toarray()
labels = df.Class


X_train,X_test,y_train,y_test = train_test_split(df['Comment'],df['Class'], random_state=0)

count_vect = CountVectorizer()

X_train_counts=count_vect.fit_transform(X_train)
X_test_cv = count_vect.transform(X_test)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB(alpha=1).fit(X_train_tfidf, y_train)

predictions = clf.predict(X_test_cv)

print('Naive Bayes score: ',accuracy_score(y_test, predictions))

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
cv_df.groupby('model_name').accuracy.mean()


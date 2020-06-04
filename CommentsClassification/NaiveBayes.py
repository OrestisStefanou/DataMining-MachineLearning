# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:29:45 2020

@author: orestis
"""

from pandas import DataFrame,read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

import pandas as pd

TrainDataLocation = './train.csv'
TestDataLocation = './test.csv'

train_df = pd.read_csv(TrainDataLocation,header=None,names=['Comment','Class'])
test_df = pd.read_csv(TestDataLocation,header=None,names=['Comment','Class'])

train_comments = train_df['Comment']
test_comments = test_df['Comment']

train_classes=train_df['Class']

#Create a Gaussian Naive Bayes model
model = GaussianNB()

#Create Train Count Vectorizer
TrainCount_Vectorizer = list()

vectorizer = CountVectorizer()
predicted_classes=list()
#Make the predictions
for test_comment in test_comments:
    TrainCount_Vectorizer=[]
    #tokenize and build vocabulary
    vectorizer.fit([test_comment])
    for x in train_comments:
        comment = [x]
        #encode the document
        vector = vectorizer.transform(comment)
        temp_array = vector.toarray()
        TrainCount_Vectorizer.append(temp_array[0])
    
    #Predict Output
    model.fit(TrainCount_Vectorizer,train_classes)
    vector = vectorizer.transform([test_comment])
    predicted = model.predict(vector.toarray())
    predicted_classes.append(predicted)

offensive = 0
neutral = 0
for class_value in predicted_classes:
    if class_value==1:
        offensive+=1
    if class_value==0:
        neutral+=1

print('Offensive comments:',offensive)
print('Neutral comments:',neutral)
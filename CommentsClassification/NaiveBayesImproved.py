
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:29:23 2020

@author: orestis
"""

from pandas import DataFrame,read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd

#Function to get accuracy of the predicted results
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(predicted)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

TrainDataLocation = './train.csv'
TestDataLocation = './test.csv'

train_df = pd.read_csv(TrainDataLocation,header=None,names=['Comment','Class'])
test_df = pd.read_csv(TestDataLocation,header=None,names=['Comment','Class'])

train_comments = train_df['Comment']
test_comments = test_df['Comment']

train_classes=train_df['Class']
test_classes=test_df['Class']

lemmatizer = WordNetLemmatizer()
#Create a set of stopwords
stop_words = set(stopwords.words('english'))

#Create a new comment train dataset
#After lemmazation and removing stopwords
cleaned_train_comments = list()
for comment in train_comments:
    example_sent = comment
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [] 
      
    for w in word_tokens: 
        if w not in stop_words: 
            lemmatizer.lemmatize(w)
            filtered_sentence.append(w) 
    
    new_sentence=''
    for word in filtered_sentence:
        new_sentence = new_sentence + word + ' '
    
    cleaned_train_comments.append(new_sentence)

#Do the same for the test comments dataset
cleaned_test_comments=list()
for comment in test_comments:
    example_sent = comment
    word_tokens = word_tokenize(example_sent)
    filtered_sentence = [] 
      
    for w in word_tokens: 
        if w not in stop_words: 
            lemmatizer.lemmatize(w)
            filtered_sentence.append(w) 
    
    new_sentence=''
    for word in filtered_sentence:
        new_sentence = new_sentence + word + ' '
    
    cleaned_test_comments.append(new_sentence)


#Create a Gaussian Naive Bayes model
model = GaussianNB()

#Create Train Count Vectorizer
TrainCount_Vectorizer = list()

vectorizer = CountVectorizer()
predicted_classes=list()
#Make the predictions
for test_comment in cleaned_test_comments:
    TrainCount_Vectorizer=[]
    #tokenize and build vocabulary
    try:
        vectorizer.fit([test_comment])
    except:
         continue   
    for x in cleaned_train_comments:
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

score = accuracy_metric(test_classes, predicted_classes)
print('Score:',score)
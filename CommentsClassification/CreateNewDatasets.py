# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:25:19 2020

@author: orestis
"""

from pandas import DataFrame,read_csv

import pandas as pd
import nltk
import string

#Location of train data
TrainLocation = '/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/CommentsClassification/data/train.csv'
#Location of test data
TestLocation = '/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/CommentsClassification/data/impermium_verification_labels.csv'

train_df = pd.read_csv(TrainLocation,names=['Class','Date','Comment'],skiprows=1)   #Get the training data
test_df = pd.read_csv(TestLocation,names=['id','Class','Date','Comment','Usage'],skiprows=1)     #Get the testing data

#Delete date column
del train_df['Date']
del test_df['Date']
#Delete id,usage column from test data
del test_df['id']
del test_df['Usage']

train_comments = train_df['Comment']
test_comments = test_df['Comment']


#PREPROCESS COMMENTS DATA
alphabet = list(string.ascii_lowercase)
numbers=['1','2','3','4','5','6','7','8','9','0',' ']
alphabet = alphabet + numbers
#Convert to lower case
for i in range(len(train_comments)):
    train_comments[i]=train_comments[i].lower()
    train_comments[i] = train_comments[i].replace('-',' ')
    train_comments[i] = train_comments[i].replace('\\',' ')
    train_comments[i] = train_comments[i].replace('.',' ')
    train_comments[i] = train_comments[i].replace(',',' ')
    train_comments[i] = train_comments[i].replace('_',' ')
    #Keep only the letters and numbers
    for c in train_comments[i]:
        if c not in alphabet:
             train_comments[i]=train_comments[i].replace(c,'')


for i in range(len(test_comments)):
    test_comments[i]=test_comments[i].lower()
    test_comments[i] = test_comments[i].replace('-',' ')
    test_comments[i] = test_comments[i].replace('\\',' ')
    test_comments[i] = test_comments[i].replace('.',' ')
    test_comments[i] = test_comments[i].replace(',',' ')
    test_comments[i] = test_comments[i].replace('_',' ')
    #Keep only the letters and numbers
    for c in test_comments[i]:
        if c not in alphabet:
             test_comments[i]=test_comments[i].replace(c,'')
    
train_class_values = train_df['Class']
test_class_values = test_df['Class']

#Create the new data sets
NewTrainDataSet = list(zip(train_comments,train_class_values))
NewTestDataSet = list(zip(test_comments,test_class_values))

new_train_df = pd.DataFrame(data=NewTrainDataSet,columns=['Comments','Classs'])
new_test_df = pd.DataFrame(data=NewTestDataSet,columns=['Comments','Class'])

Dataset = NewTrainDataSet + NewTestDataSet
df = pd.DataFrame(data=Dataset,columns=['Comments','Class']) 

new_train_df.to_csv('train.csv',index=False,header=False)
new_test_df.to_csv('test.csv',index=False,header=False)
df.to_csv('data.csv',index=False,header=False)
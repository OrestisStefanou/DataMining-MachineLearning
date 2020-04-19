#!/usr/bin/env python3

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import os
import string
import numpy as np
import pandas as pd
import re
import linecache
import operator



def find_class(score_list,neighbor_num):
    score_dict = {'sport':0,'tech':0,'politics':0,'entertainment':0,'business':0}
    for i in range(0,neighbor_num):
        category = score_list[i][1]
        score_dict[category]+=1
    return max(score_dict.items(), key=operator.itemgetter(1))[0]



vectorizer = TfidfVectorizer()

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


train_df=pd.DataFrame()

Location = r'/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/articles_project/train_set.csv'
train_df=pd.read_csv(Location,header=None,names=['id','title','content','category'])

test_df=pd.DataFrame()

Location = r'/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/articles_project/test_set.csv'
test_df=pd.read_csv(Location,header=None,names=['id','title','content','category'])


train_article_categories = train_df['category']

train_articles_keywords = [] #A list with the keywords of every train article,and their category
f = open('train_keywords.txt','r')
index = 0
for line in f:
    train_articles_keywords.append((line,train_article_categories[index]))
    index+=1
f.close()

test_articles_keywords = [] #A list with the keywords of every test article
f=open('test_keywords.txt','r')
for line in f:
    test_articles_keywords.append(line)
f.close()

predictions = []
correct_predictions = test_df['category']
counter = 0
#Make a testing prediction
#test_article_keyword = test_articles_keywords[150]
for test_article_keyword in test_articles_keywords:
    if(counter==10):    #Checkari mono ta prota 10 giati argi polla
        break
    cosine_sim_scores=[]

    for keywords in train_articles_keywords:    #Compare test article's keywords with train articles keywords
        cosine_score = cosine_sim(test_article_keyword,keywords[0])
        cosine_sim_scores.append((cosine_score,keywords[1]))

    #print(test_article_keyword)
    cosine_sim_scores.sort(key = operator.itemgetter(0), reverse = True)

    predictions.append(find_class(cosine_sim_scores,5))
    counter+=1

correct = 0 #counter of correct predictions
wrong = 0   #counter of wrong predictions
for i in range(0,len(predictions)):
    if(predictions[i]==correct_predictions[i]):
        correct+=1
    else:
        wrong+=1

print('Correct predictions: ',correct)
print('Wrong predictions: ',wrong)
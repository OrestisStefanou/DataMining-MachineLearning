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
import math
import linecache
import operator


vectorizer = TfidfVectorizer()

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


def get_id_index(ids,id):#Get the index of the target id(index in the lists and number of line in keywords file)
    for i in range(0,len(ids)):
        if(str(ids[i])==str(id)):
            return i+1



def recommend(target_id,number):
    Location = r'/home/orestis/EKPA/6examino/DataMining/project1/train.csv'
    df=pd.read_csv(Location)

    ids = df['id']
    names = df['name']
    descriptions = df['description']

    target_index = get_id_index(ids,target_id)

    target_keywords = linecache.getline('keywords.txt',target_index)#Get the keywords of target
    cosine_sim_array = []#An aray (with cosine similarity and index) of target with all other keywords
    keywords_file = open("keywords.txt","r")
    for i in range(10):
        if i != target_index and ids[i]!=ids[target_index]:
            keywords = keywords_file.readline()
            cosine_sim_array.append((cosine_sim(target_keywords,keywords),i))#append a tuple of cosine similarity and index because we will sort valus later
        else:
            cosine_sim_array.append((0,i))

    cosine_sim_array.sort(key = operator.itemgetter(0), reverse = True)
    print('Recommending ' + str(number) + ' listings similiar to id:' + str(target_id))
    for i in range(0,number):
        index = cosine_sim_array[i][1]#get index
        score = cosine_sim_array[i][0] 
        print('Recommended:' + str(names[index]))    
        print('Description:' + str(descriptions[index]))
        print("(score:"+str(score)+")")


recommend(10595,2)

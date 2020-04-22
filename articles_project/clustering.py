#!/usr/bin/env python3

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
#import mpld3

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


Location = r'/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/articles_project/train_set.csv'
train_df=pd.read_csv(Location,header=None,names=['id','title','content','category'])

test_df=pd.DataFrame()

Location = r'/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/articles_project/test_set.csv'
test_df=pd.read_csv(Location,header=None,names=['id','title','content','category'])

test_df = test_df.drop(['category'],axis=1)

titles = train_df['title']
synopses = train_df['content']
categories = train_df['category']

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

articles = { 'title': titles,'content': synopses, 'cluster': clusters }
#frame = pd.DataFrame(articles, index = [clusters] , columns = ['title', 'cluster'])
frame = pd.DataFrame(articles , columns = ['title', 'cluster'])

print(frame['cluster'].value_counts()) #number of articles per cluster (clusters from 0 to 4)

counter = 0
for x in frame['cluster']:
    print('Clustering category:'+str(x) + '     Correct Category:' + str(categories[counter]))
    counter+=1
    #if counter==5:
    #    break


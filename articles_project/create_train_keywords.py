#!/usr/bin/env python3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import os
import numpy as np
import pandas as pd
import re

def preprocess(text):
    #lowercase
    text = (str(text).lower())
    #remove tags
    text = re.sub("<!--?.*?-->","",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text


def get_stop_words():
    return frozenset(stopwords.words('english'))



def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)



def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results




df=pd.DataFrame()

Location = r'/home/orestis/Desktop/GitHubRepositories/DataMining-MachineLearning/articles_project/train_set.csv'
df=pd.read_csv(Location,header=None,names=['id','title','content','category'])

df['text'] = df['title'] + df['content']
df['text'] = df['text'].apply(lambda x:preprocess(x))

stopwords = get_stop_words()

docs = df['text'].tolist()

cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
 
# you only needs to do this once, this is a mapping of index to 
feature_names=cv.get_feature_names()


keywords_list = []
# get the document that we want to extract keywords from
for doc in docs:
    #generate tf-idf for the given document
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())

    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,20)
    keywords_list.append(list(keywords.keys()))

print(keywords_list[0])
print(keywords_list[1])

#SAVE KEYWORDS TO A FILE
f = open('train_keywords.txt','w')
for words in keywords_list:
    for word in words:
        f.write(word)
        f.write(' ')
    f.write('\n')
f.close()
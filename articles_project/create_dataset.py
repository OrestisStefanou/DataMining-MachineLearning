#!/usr/bin/env python3
import pandas as pd
import os
from pandas import DataFrame, read_csv

sport_path = '/home/orestis/EKPA/6examino/DataMining/fulltext/data/sport/'
tech_path = '/home/orestis/EKPA/6examino/DataMining/fulltext/data/tech/'
politics_path = '/home/orestis/EKPA/6examino/DataMining/fulltext/data/politics/'
entert_path = '/home/orestis/EKPA/6examino/DataMining/fulltext/data/entertainment/'
business_path = '/home/orestis/EKPA/6examino/DataMining/fulltext/data/business/'

#FOR TRAIN_SET FILE
ids = []    #A list with id of each article
titles = [] #A list with the title of each article
contents = []   #A list with the content of each article
categories = [] #A list with the category of each article

#FOR TEST_SET FILE
test_ids = []    #A list with id of each article
test_titles = [] #A list with the title of each article
test_contents = []   #A list with the content of each article
test_categories = [] #A list with the category of each article

test_article_id=0   #Article id for test set
article_id = 0  #article id for train set
counter = 0     #a counter to check if we reached 80% of folder's documents
#GO THROUGH EACH FILE IN sport_path
for filename in os.listdir(sport_path):
    location = sport_path + filename
    f=open(location,'r')    #Open the file
    try:
        title = f.readline()    #First line is the title
    except:
        continue    #i have problem reading some files
    
    if(counter<int(len(os.listdir(sport_path)) * 0.8)):
        ids.append(article_id)  #Add the id to the list
        titles.append(str(title))    #Add the title to the list
        categories.append('sport')  #Add the category to the list
        article_id+=1           #For the next article
        contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1
    else:
        test_ids.append(test_article_id)  #Add the id to the list
        test_titles.append(str(title))    #Add the title to the list
        test_categories.append('sport')  #Add the category to the list
        test_article_id+=1           #For the next article
        test_contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1

    f.close()

counter = 0
#GO THROUGH EACH FILE IN tech_path
for filename in os.listdir(tech_path):
    location = tech_path + filename
    f=open(location,'r')    #Open the file
    try:
        title = f.readline()    #First line is the title
    except:
        continue    #i have problem reading some files
    
    if(counter<int(len(os.listdir(tech_path)) * 0.8)):
        ids.append(article_id)  #Add the id to the list
        titles.append(str(title))    #Add the title to the list
        categories.append('tech')  #Add the category to the list
        article_id+=1           #For the next article
        contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1
    else:
        test_ids.append(test_article_id)  #Add the id to the list
        test_titles.append(str(title))    #Add the title to the list
        test_categories.append('tech')  #Add the category to the list
        test_article_id+=1           #For the next article
        test_contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1

    f.close()

counter = 0
#GO THROUGH EACH FILE IN politics_path
for filename in os.listdir(politics_path):
    location = politics_path + filename
    f=open(location,'r')    #Open the file
    try:
        title = f.readline()    #First line is the title
    except:
        continue    #i have problem reading some files
    
    if(counter<int(len(os.listdir(politics_path)) * 0.8)):
        ids.append(article_id)  #Add the id to the list
        titles.append(str(title))    #Add the title to the list
        categories.append('politics')  #Add the category to the list
        article_id+=1           #For the next article
        contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1
    else:
        test_ids.append(test_article_id)  #Add the id to the list
        test_titles.append(str(title))    #Add the title to the list
        test_categories.append('politics')  #Add the category to the list
        test_article_id+=1           #For the next article
        test_contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1

    f.close()

counter = 0
#GO THROUGH EACH FILE IN entert_path
for filename in os.listdir(entert_path):
    location = entert_path + filename
    f=open(location,'r')    #Open the file
    try:
        title = f.readline()    #First line is the title
    except:
        continue    #i have problem reading some files
    
    if(counter<int(len(os.listdir(entert_path)) * 0.8)):
        ids.append(test_article_id)  #Add the id to the list
        titles.append(str(title))    #Add the title to the list
        categories.append('entertainment')  #Add the category to the list
        article_id+=1           #For the next article
        contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1
    else:
        test_ids.append(article_id)  #Add the id to the list
        test_titles.append(str(title))    #Add the title to the list
        test_categories.append('entertainment')  #Add the category to the list
        test_article_id+=1           #For the next article
        test_contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1

    f.close()

#GO THROUGH EACH FILE IN business_path
for filename in os.listdir(business_path):
    location = business_path + filename
    f=open(location,'r')    #Open the file
    try:
        title = f.readline()    #First line is the title
    except:
        continue    #i have problem reading some files
    
    if(counter<int(len(os.listdir(business_path)) * 0.8)):
        ids.append(article_id)  #Add the id to the list
        titles.append(str(title))    #Add the title to the list
        categories.append('business')  #Add the category to the list
        article_id+=1           #For the next article
        contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1
    else:
        test_ids.append(test_article_id)  #Add the id to the list
        test_titles.append(str(title))    #Add the title to the list
        test_categories.append('business')  #Add the category to the list
        test_article_id+=1           #For the next article
        test_contents.append(str(f.read()))   #Add the rest of article in the content list
        counter+=1

    f.close()

TrainDataSet = list(zip(ids,titles,contents,categories))
train_df = pd.DataFrame(data=TrainDataSet,columns=['id','title','content','category'])

TestDataSet = list(zip(test_ids,test_titles,test_contents))
test_df = pd.DataFrame(data=TestDataSet,columns=['id','title','content'])

train_df.to_csv('train_set.csv',index=False,header=False)
test_df.to_csv('test_set.csv',index=False,header=False)
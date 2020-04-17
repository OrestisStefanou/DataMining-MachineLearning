#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 11:59:27 2020

@author: orestis
"""

from pandas import DataFrame,read_csv

import matplotlib.pyplot as plt
import pandas as pd
import sys
import matplotlib

columns=['id','zipcode','transit','bedrooms','beds','review_scores_rating','number_of_reviews','neighbourhood_cleansed',
         'neighbourhood_group_cleansed','name','latitude','longitude','last_review','instant_bookable','host_since',
         'host_response_rate','host_identity_verified','host_has_profile_pic','first_review','description','city',
         'cancellation_policy','bed_type','bathrooms','accommodates','amenities','room_type','property_type','price',
         'availability_365','minimum_nights']

april_df=pd.DataFrame()
Location = r'/home/orestis/EKPA/6examino/DataMining/data/april/listings.csv'
april_df=pd.read_csv(Location,low_memory=False)


for x in april_df:
    if x not in columns:
        del april_df[x]

#add month column
april_df['Month'] = 'April'

febrouary_df=pd.DataFrame()
Location = r'/home/orestis/EKPA/6examino/DataMining/data/febrouary/listings.csv'
febrouary_df=pd.read_csv(Location,low_memory=False)

for x in febrouary_df:
    if x not in columns:
        del febrouary_df[x]


#add month column
febrouary_df['Month'] = 'Febrouary'

march_df=pd.DataFrame()
Location = r'/home/orestis/EKPA/6examino/DataMining/data/march/listings.csv'
march_df=pd.read_csv(Location, low_memory=False)

for x in march_df:
    if x not in columns:
        del march_df[x]

#add month column
march_df['Month'] = 'March'

df = pd.concat([april_df,febrouary_df,march_df])

for x in df:
    print(x)

all_ids = df['id']
id_unique = df['id'].unique()

df.to_csv('train.csv')
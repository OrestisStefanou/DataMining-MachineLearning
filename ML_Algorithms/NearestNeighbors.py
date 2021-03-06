# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:06:50 2020

@author: orestis
"""
from math import sqrt
from random import seed
from random import randrange
from csv import reader

#Load a csv file
def load_csv(filename):
    dataset = list()
    with open(filename,'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

#Convert string column to float
def str_column_to_float(dataset,column):
    for row in dataset:
        try:
            row[column] = float(row[column].strip())
        except:
            continue
    
#Convert string column to integer
def str_column_to_int(dataset,column):
    class_values = [row[column] for row in dataset]
    unique=set(class_values)
    lookup = dict()
    for i,value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    #return lookup


class KNN_Classification:
    def __init__(self,filename):
        self.dataset = load_csv(filename)
        for i in range(len(self.dataset)):
            del self.dataset[i][0]
        
        for i in range(len(self.dataset[0])-1):
        	str_column_to_float(self.dataset, i)
        # convert class column to integers
        str_column_to_int(self.dataset, len(self.dataset[0])-1)

    # Split a dataset into k folds
    def cross_validation_split(self,dataset, n_folds):
    	dataset_split = list()
    	dataset_copy = list(dataset)
    	fold_size = int(len(dataset) / n_folds)
    	for _ in range(n_folds):
    		fold = list()
    		while len(fold) < fold_size:
    			index = randrange(len(dataset_copy))
    			fold.append(dataset_copy.pop(index))
    		dataset_split.append(fold)
    	return dataset_split
     
    # Calculate accuracy percentage
    def accuracy_metric(self,actual, predicted):
    	correct = 0
    	for i in range(len(actual)):
    		if actual[i] == predicted[i]:
    			correct += 1
    	return correct / float(len(actual)) * 100.0
     
    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self,dataset, n_folds, *args):
    	folds = self.cross_validation_split(dataset, n_folds)
    	scores = list()
    	for fold in folds:
    		train_set = list(folds)
    		train_set.remove(fold)
    		train_set = sum(train_set, [])
    		test_set = list()
    		for row in fold:
    			row_copy = list(row)
    			test_set.append(row_copy)
    			row_copy[-1] = None
    		predicted = self.k_nearest_neighbors(train_set, test_set, *args)
    		actual = [row[-1] for row in fold]
    		accuracy = self.accuracy_metric(actual, predicted)
    		scores.append(accuracy)
    	return scores
    
    
    def euclidean_distance(self,row1,row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)
        
    
    #Find most similiar neigbors
    def get_neighbors(self,train,test_row,k):
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row,dist))
        distances.sort(key=lambda tup:tup[1])
        neighbors = list()
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors
    
    #Make prediction
    def predict(self,train,test_row,k):
        neighbors = self.get_neighbors(train, test_row, k)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values),key=output_values.count)
        return prediction
    
    # kNN Algorithm
    def k_nearest_neighbors(self,train, test, num_neighbors):
    	predictions = list()
    	for row in test:
    		output = self.predict(train, row, num_neighbors)
    		predictions.append(output)
    	return(predictions)
    
    def evaluation(self,k,folds_num):
        scores = self.evaluate_algorithm(self.dataset,folds_num,k)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))        


class KNN_Recomendation:
    def __init__(self,filename):
        self.filename = filename
        dataset = load_csv(filename)
        column_names = dataset[0]   #Get column names
        del dataset[0]  #Delete the first row that contains the column names
        needed_columns = ['budget','genres','keywords','title','vote_average'] #Columns that we need
        #Get the indexes of the columns we need
        needed_indexes = list()
        for i in range(len(column_names)):
            if column_names[i] in needed_columns:
                needed_indexes.append(i)
                
        self.newDataset = list()
        #Create a new dataset with only the columns we need
        for i in range(len(dataset)):
            newrow=[]
            for x in needed_indexes:
                newrow.append(dataset[i][x])
            self.newDataset.append(newrow)
        
        #Delete the old one
        del dataset
        
        #Create a dictionary with the index of the column names
        columns_index=dict()
        for i in range(len(needed_columns)):
            columns_index[needed_columns[i]]=i
        
        numbers = ['0','1','2','3','4','5','6','7','8','9']
        
        #Convert budget column to float
        str_column_to_float(self.newDataset,0)
        #Convert vote_average column to float
        str_column_to_float(self.newDataset,4)
        
        genres_ints = list()
        #Transform genres column to a list of ints
        for i in range(len(self.newDataset)):
            genres_ints=[]
            genres = self.newDataset[i][1].split(',')
            for x in genres:
                for c in x:
                    if c not in numbers:
                        x = x.replace(c, '')
                if len(x)>0:
                    genres_ints.append(x)
            for j in range(len(genres_ints)):
                genres_ints[j] = int(genres_ints[j])
            self.newDataset[i][1] = genres_ints
        
        keywords_int=list()
        #Transform keywords column to a list of ints
        for i in range(len(self.newDataset)):
            keywords_int=[]
            keywords = self.newDataset[i][2].split(',')
            for x in keywords:
                for c in x:
                    if c not in numbers:
                        x = x.replace(c, '')
                if len(x)>0:
                    keywords_int.append(x)
            for j in range(len(keywords_int)):
                keywords_int[j] = int(keywords_int[j])
            self.newDataset[i][2] = keywords_int

    #Calculate distance between 2 movies
    def distance(self,row1,row2):
        distance = 0.0
        distance += (row1[0] - row2[0])**2
        same_genres_dist=len(row1[1])
        for x in row1[1]:
            if x in row2[2]:
                if(same_genres_dist>0):
                    same_genres_dist-=1
                else:
                    break
        distance += (same_genres_dist)**2
        
        keywords_dist=10
        for x in row1[2]:
            if x in row2[2]:
                if(keywords_dist>0):
                    keywords_dist-=1
                else:
                    break
        distance+=(keywords_dist)**2
        distance += (row1[4] - row2[4])**2
        return sqrt(distance)

    #Find most similiar movies
    def get_similiar_movies(self,train,test_row,k):
        distances = list()
        for train_row in train:
            if(train_row[3]==test_row[3]):  #If it's the same movie ignore it
                continue
            dist = self.distance(test_row, train_row)
            distances.append((train_row,dist))
        distances.sort(key=lambda tup:tup[1])
        neighbors = list()
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

    #Recomendation
    def recommendation(self,index,k):
        test_row = self.newDataset[index]
        neigbors=self.get_similiar_movies(self.newDataset, test_row, k)
        output_values = [row[3] for row in neigbors]
        print('Recommended movies if you liked ' + test_row[3])
        for movie in output_values:
            print(movie)
  
# Test the kNN on the Iris Flowers dataset
seed(1)
filename='iris.csv'
classifier = KNN_Classification(filename)
classifier.evaluation(10, 5)

filename = 'movies.csv'

movie_rec = KNN_Recomendation(filename)
movie_rec.recommendation(10,10)
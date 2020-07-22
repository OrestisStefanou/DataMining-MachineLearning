from math import sqrt
from random import randrange
from random import seed
from csv import reader

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 


class LVQ():
    # Split a dataset into k folds
    def cross_validation_split(self,dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    def euclidean_distance(self,row1,row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)
    #Locate the best matching unit
    def get_best_matching_unit(self,codebooks,test_row):
        distances = list()
        for codebook in codebooks:
            dist = self.euclidean_distance(codebook,test_row)
            distances.append((codebook,dist))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]

    #Create a random codebook vector
    def random_codebook(self,train):
        n_records = len(train)
        n_feautures = len(train[0])
        codebook = [train[randrange(n_records)][i] for i in range(n_feautures)]
        return codebook
    #Train a set of codebook vectors
    def train_codebooks(self,train,n_codebooks,lrate,epochs):
        codebooks = [self.random_codebook(train) for i in range(n_codebooks)]
        for epoch in range(epochs):
            rate = lrate * (1.0-(epoch/float(epochs)))
            sum_error = 0.0
            for row in train:
                bmu = self.get_best_matching_unit(codebooks,row)
                for i in range(len(row)-1):
                    error = row[i] - bmu[i]
                    sum_error += error**2
                    if bmu[-1] == row[-1]:
                        bmu[i] += rate * error
                    else:
                        bmu[i] -= rate * error
            #print('>epoch=%d,lrate=%.3f,error=%.3f' % (epoch,rate,sum_error))
        return codebooks

    # Calculate accuracy percentage
    def accuracy_metric(train,actual, predicted):
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
            predicted = self.lvq(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    #Make a prediction with codebook vectors
    def predict(self,codebooks,test_row):
        bmu = self.get_best_matching_unit(codebooks,test_row)
        return bmu[-1]
    
    #LVQ Algorithm
    def lvq(self,train,test,n_codebooks,lrate,epochs):
        codebooks = self.train_codebooks(train,n_codebooks,lrate,epochs)
        predictions = list()
        for row in test:
            output = self.predict(codebooks,row)
            predictions.append(output)
        return(predictions)


seed(1)
lvq = LVQ()
filename = 'ionosphere.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# evaluate algorithm
n_folds = 5
learn_rate = 0.3
n_epochs = 50
n_codebooks = 20
scores = lvq.evaluate_algorithm(dataset, n_folds, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
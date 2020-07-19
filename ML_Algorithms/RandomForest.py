from random import seed
from random import randrange
from csv import reader
from math import sqrt

#Load a CSV file
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
        row[column] = float(row[column].strip())


#Convert string column to integer
def str_column_to_int(dataset,column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i,value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup   #Check why we return this


class RandomForest:
    def __init__(self):
        print("Random Forest Classifier created")
    #Split a dataset into k folds
    def cross_validation_script(self,dataset,n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset)/n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    #Calculate accuracy percentage
    def accuracy_metric(self,actual,predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct/float(len(actual)) * 100.0
    

    #Split a dataset based on an attribute and an attribute value
    def test_split(self,index,value,dataset):
        left,right = list(),list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left,right

    #Calculate the Gini index for a split dataset
    def gini_index(self,groups,classes):
        #count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        #sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            #avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            #score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val)/size
                score += p * p
            #weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini
    

    #Select the best split point for a dataset
    def get_split(self,dataset,n_features):
        class_values = list(set(row[-1] for row in dataset))
        b_index,b_value,b_score,b_groups = 999,999,999,None
        features = list()
        #Create a list with random indexes(features)
        while len(features) < n_features:
            index = randrange(len(dataset[0])-1)
            if index not in features:
                features.append(index)
        
        for index in features:
            for row in dataset:
                groups = self.test_split(index,row[index],dataset)
                gini = self.gini_index(groups,class_values)
                if gini < b_score:
                    b_index,b_value,b_score,b_groups = index,row[index],gini,groups
        return {'index':b_index,'value':b_value,'groups':b_groups}            

    #Create a terminal node value
    def to_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key = outcomes.count)
    
    #Check if most of the records in the group belong in the same class
    def check_class_percentage(self,group):
        all_class_values = [row[-1] for row in group]
        classes_sum = dict()
        for x in all_class_values:
            try:
                classes_sum[x]+=1
            except:
                classes_sum[x]=1
        
        for x in classes_sum.values():
            if float(100/(len(all_class_values)/x)) > 98.0: #if 98% of records belong in the same class make it a terminal node
                return True
            else:
                return False

    #Create child splits for a node or make terminal
    def split(self,node,max_depth,min_size,n_features,depth):
        left,right = node['groups']
        del(node['groups'])
        #check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        #check for max depth
        if depth >= max_depth:
            node['left'],node['right'] = self.to_terminal(left),self.to_terminal(right)
            return
        #process left child
        if len(left) <= min_size or self.check_class_percentage(left):
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left,n_features)
            self.split(node['left'],max_depth,min_size,n_features,depth+1)
        #process right child
        if len(right) <= min_size or self.check_class_percentage(right):
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right,n_features)
            self.split(node['right'],max_depth,min_size,n_features,depth+1)
    
    #Make a prediction with a decision tree
    def predict(self,node,row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'],dict):
                return self.predict(node['left'],row)
            else:
                return node['left']
        else:
            if isinstance(node['right'],dict):
                return self.predict(node['right'],row)
            else:
                return node['right']
    
    #Create a random subsample from the dataset with replacement
    def subsample(self,dataset,ratio):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            sample.append(dataset[index])
        return sample

    #Build a decision tree
    def build_tree(self,train,max_depth,min_size,n_features):
        root = self.get_split(train,n_features)
        self.split(root,max_depth,min_size,n_features,1)
        return root

    # Make a prediction with a list of bagged trees
    def bagging_predict(self,trees, row):
        predictions = [self.predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    #Random Forest Algorithm
    def random_forest(self,train,test,max_depth,min_size,sample_size,n_trees,n_features):
        trees = list()
        for i in range(n_trees):
            sample = self.subsample(train,sample_size)
            tree = self.build_tree(sample,max_depth,min_size,n_features)
            trees.append(tree)
        predictions = [self.bagging_predict(trees,row) for row in test]
        return(predictions)

    #Evaluate the algorithm
    def evaluate_algorithm(self,dataset,n_folds,*args):
        folds = self.cross_validation_script(dataset,n_folds)
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
            predicted = self.random_forest(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores    

#Test the algorithm
seed(2)
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(0,len(dataset[0])-1):
    str_column_to_float(dataset,i)
str_column_to_int(dataset,len(dataset[0])-1)

random_forest = RandomForest()
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1,5,10]:
    scores = random_forest.evaluate_algorithm(dataset,n_folds,max_depth,min_size,sample_size,n_trees,n_features)
    print('Trees: %d' % n_trees)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))   

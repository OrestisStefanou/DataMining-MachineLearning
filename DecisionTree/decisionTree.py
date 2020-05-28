from random import seed
from random import randrange
from csv import reader


class DecisionTree:
    def __init__(self,dataset_filename,min_size,max_depth):
        self.max_depth = max_depth
        self.min_size = min_size
        self.dataset = self.load_csv(dataset_filename)
        for i in range(len(self.dataset[0])):
            self.str_column_to_float(self.dataset,i)
        
    
    def load_csv(self,filename):
        file = open(filename,"rt")
        lines = reader(file,delimiter="\t")
        dataset = list(lines)   #Each element of the list is a record
        return dataset

    #Convert records data from string to float
    def str_column_to_float(self,dataset,column):
        for row in dataset:
            try:
                row[column] = float(row[column].strip())
            except:
                continue

    #Split a dataset into k folds
    def cross_validation_split(self,dataset,n_folds):
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
                correct+=1
        return correct / float(len(actual))*100.0


    #Evaluate the algorithm using a cross validation script
    def evaluation(self,dataset,n_folds,*args):
        folds = self.cross_validation_split(dataset,n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set,[])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = self.decision_tree(train_set,test_set)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual,predicted)
            scores.append(accuracy)
        return scores


    #Calculate the Gini index for a split dataset
    def gini_index(self,groups,classes):
        #count all samples at split point
        n_instances=float(sum([len(group) for group in groups]))
        #sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            #avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            #score the group bases on the score for each class
            for class_val in classes:
                p=[row[-1] for row in group].count(class_val)/size
                score += p*p 
            #weight the group score by its relative size
            gini+=(1.0-score) * (size /n_instances)
        return gini


    #Split a dataset based on an attribute and an attribute value
    def test_split(self,index,value,dataset):
        left,right = list(),list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left,right


    #Select the best split point for a dataset
    def get_split(self,dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index,b_value,b_score,b_groups = 999,999,999,None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index,row[index],dataset)
                gini = self.gini_index(groups,class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}


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
            if float(100/(len(all_class_values)/x)) > 99.0: #if 99% of records belong in the same class make it a terminal node
                return True

    #Create a terminal node value
    def make_terminal(self,group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes),key = outcomes.count)


    #Create child splits for a node or make terminal
    def split(self,node,depth):
        left,right = node['groups']
        del(node['groups'])
        #check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.make_terminal(left + right)
            return
        #check for max depth
        #if depth >=self.max_depth:
        #    node['left'],node['right'] = self.make_terminal(left),self.make_terminal(right)
        #    return
        #process left child
        if len(left)<=self.min_size or self.check_class_percentage(left):
            node['left'] = self.make_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'],depth+1)
        #process right child
        if len(right) <= self.min_size or self.check_class_percentage(right):
            node['right'] = self.make_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'],depth+1)

    #Build the decision tree
    def build_tree(self,train):
        root = self.get_split(train)
        self.split(root,1)
        return root

    # Print the decision tree
    def print_tree(self,node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))


    #Make a prediction
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


    #Classification and Regression Tree Algorithm
    def decision_tree(self,train,test):
        tree = self.build_tree(train)
        predictions = list()
        for row in test:
            prediction = self.predict(tree,row)
            predictions.append(prediction)
        return (predictions)
    
    #Evaluate the algorithm
    def evaluate(self,folds_num):
        scores = self.evaluation(self.dataset,folds_num)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))



seed(1)
# load and prepare data
filename = 'PhishingData.csv'
n_folds = 5
max_depth = 5
min_size = 10

decision_tree = DecisionTree(filename,min_size,max_depth)
decision_tree.evaluate(n_folds)
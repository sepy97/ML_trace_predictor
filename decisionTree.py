from cmath import inf
import numpy as np
from sklearn import tree
from sklearn import ensemble 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class decisionTree:
    def __init__(self, num_classes, max_depth = 10, num_min_split = 2, num_min_leaf = 1, num_features = None):
        self.classes = np.arange(0, num_classes, 1) 
        self.clf = tree.DecisionTreeClassifier()
        self.depth = max_depth
        self.min_samples_split = num_min_split
        self.min_samples_leaf = num_min_leaf
        self.max_features = num_features
        self.criterion = 'entropy'
    #def predict_and_update
    
    def predict(self, data_point):
        pred = self.clf.predict(data_point);
        return pred

    def fit(self, data_point, label): 
        params = {'criterion' : self.criterion, 'max_depth' : self.depth, 'min_samples_split' : self.min_samples_split, 'min_samples_leaf' : self.min_samples_leaf}
        self.clf.set_params(**params)
        self.clf.fit(data_point, label)
        return 1 - self.clf.score(data_point, label)

class randomForest:
    def __init__(self, num_classes, num_estimator, max_depth, num_min_split, num_min_leaf, num_features = None):
        self.classes = np.arange(0, num_classes, 1) 
        self.clf = ensemble.RandomForestClassifier()
        self.depth = max_depth;
        self.n_estimators = num_estimator;
        self.max_features = num_features;
        self.min_samples_split = num_min_split;
        self.min_samples_leaf = num_min_leaf;
        self.criterion = 'entropy'
    #def predict_and_update
    
    def predict(self, data_point):
        pred = self.clf.predict(data_point);
        return pred

    def fit(self, data_point, label): 
        params = {'n_estimators' : self.n_estimators, 'max_features' : self.max_features,'criterion' : self.criterion, 'max_depth' : self.depth, 'min_samples_split' : self.min_samples_split, 'min_samples_leaf' : self.min_samples_leaf}
        self.clf.set_params(**params)
        self.clf.fit(data_point, label)
        return 1 - self.clf.score(data_point, label)

class selectedDT_GridSearch:
    def __init__(self, criterion = ['entropy'], num_features = [None], num_classes = None, max_depth = 10, num_min_split = 2, num_min_leaf = 1):
        self.classes = np.arange(0, num_classes, 1) 
        self.clf = tree.DecisionTreeClassifier()        
        self.grid_search = {
            'criterion': criterion,
            'max_depth': max_depth,
            'max_features':num_features,
            'min_samples_leaf': num_min_leaf,
            'min_samples_split': num_min_split}
        self.model = GridSearchCV(estimator = self.clf, param_grid = self.grid_search, verbose= 5, n_jobs = -1) 

    def predict(self, data_point):
        pred = self.best_estimator_.predict(data_point)
        return pred

    def fit(self, data_point, label): 
        self.model.fit(data_point, label)
        return self.model

class selectedDT_RondomSearch:
    def __init__(self, criterion = ['entropy'], num_features = [None], num_classes = None, max_depth = 10, num_min_split = 2, num_min_leaf = 1):
        self.classes = np.arange(0, num_classes, 1) 
        self.clf = tree.DecisionTreeClassifier()        
        self.random_search = {'criterion': criterion,
                'max_depth': max_depth,
                'max_features': num_features,
                'min_samples_leaf': num_min_leaf,
                'min_samples_split': num_min_split}
        self.model = RandomizedSearchCV(estimator = self.clf, param_distributions = self.random_search, n_iter = 100, verbose=1, n_jobs = -1) 

    def predict(self, data_point):
        pred = self.model.best_estimator_.predict(data_point)
        return pred

    def fit(self, data_point, label): 
        self.model.fit(data_point, label)
        return self.model

class selectedRF_GridSearch:
    def __init__(self, criterion = ['entropy'], num_features = [None], num_classes = None, max_depth = 10, num_min_split = 2, num_min_leaf = 1, num_estimator = 1):
        self.classes = np.arange(0, num_classes, 1) 
        self.clf = ensemble.RandomForestClassifier()        
        self.grid_search = {
            'criterion': criterion,
            'max_depth': max_depth,
            'max_features':num_features,
            'min_samples_leaf': num_min_leaf,
            'min_samples_split': num_min_split,
            'n_estimators': num_estimator
            }
        self.model = GridSearchCV(estimator = self.clf, param_grid = self.grid_search, verbose= 1, n_jobs = -1) 

    def predict(self, data_point):
        dp = [data_point] 
        pred = self.best_estimator_.predict(dp)
        return pred

    def fit(self, data_point, label): 
        self.model.fit(data_point, label)
        return self.model

class selectedRF_RandomSearch:
    def __init__(self, criterion = ['entropy'], num_features = [None], num_classes = None, max_depth = 10, num_min_split = 2, num_min_leaf = 1, num_estimator = 1):
        self.classes = np.arange(0, num_classes, 1) 
        self.clf = ensemble.RandomForestClassifier()        
        self.random_search = {
            'criterion': criterion,
            'max_depth': max_depth,
            'max_features':num_features,
            'min_samples_leaf': num_min_leaf,
            'min_samples_split': num_min_split,
            'n_estimators': num_estimator
            }
        self.model = RandomizedSearchCV(estimator = self.clf, param_distributions = self.random_search, n_iter = 100, verbose= 1, n_jobs = -1) 
    def predict(self, data_point):
        dp = data_point 
        pred = self.model.best_estimator_.predict(data_point)
        return pred

    def fit(self, data_point, label): 
        self.model.fit(data_point, label)
        return self.model
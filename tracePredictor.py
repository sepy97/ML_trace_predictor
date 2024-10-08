import perceptron
import collections
import numpy as np
from skmultiflow.data import DataStream 
class tracePredictor:
    def __init__(self, num_features, num_classes, model):
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = model
        self.training_dataset = np.array([])
        self.training_labels = np.array([])
        self.test_dataset = np.array([])
        self.test_labels = np.array([])
        self.training_error = 0
        self.test_error = 0

    def predict(self):
        # loop through data in test dataset, call model.predict() on each data point, check if prediction matches actual value, if so, increment correct counter, increment total counter, return correct/total
        correct = 0
        total = 0
        dataset = self.test_dataset
        labels  = self.test_labels
        M,N = dataset.shape
        dataset = np.hstack((np.ones((M,1)), dataset))
        for i in range(M):
            trace_id = self.model.predict(dataset[i]) 
            if trace_id == labels[i]:
                correct += 1
            total += 1
        self.test_error = 1.0 - float(correct)/float(total)
        return self.test_error

    def train(self):
        # transfer training data to model and fit the model
        dataset = self.training_dataset
        labels  = self.training_labels
        M,N = dataset.shape
        dataset = np.hstack((np.zeros((M,1)), dataset))
        self.training_error = self.model.fit(dataset, labels)
        return self.training_error

class onlineTracePredictor:
    def __init__ (self, num_features, num_classes, training_dataset, training_labels, test_dataset, test_labels, model):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_of_datapoints = test_dataset.shape[0]
        self.model = model
        self.target_values = list(np.arange(0, num_classes, 1))
        self.datastream = DataStream(test_dataset, y=test_labels)
        self.model.partial_fit(training_dataset, training_labels, self.target_values)
        self.training_error = 0
        self.test_error = 0

    def predict_and_update(self):
        total = 0
        correct = 0
        
        while total < self.num_of_datapoints and self.datastream.has_more_samples():
            X, y = self.datastream.next_sample()
            trace_id = self.model.predict(X)
            if y == trace_id:
                correct += 1
            total += 1
            self.model.partial_fit(X, y, classes = self.target_values)

        self.test_error = 1.0 - float(correct)/float(total)
        return self.test_error

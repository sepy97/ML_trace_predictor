import numpy as np

class perceptron:
    def __init__(self, num_features, num_classes):
        self.classes = range(num_classes) 
        self.weights = np.zeros(num_features, num_classes) 
    
    def predict(self, data_point):
        # return class label (one of self.classes) based on class data point
        max_product = 0.0
        max_class = -1
        for c in self.classes:
            product = np.dot(self.weights[:,c], data_point)
            if product > max_product:
                max_product = product
                max_class = c
        return max_class

    def fit(self, data_point, label):
        # update weights based on data point and label
        max_product = 0.0
        max_class = -1
        for c in self.classes:
            product = np.dot(self.weights[:,c], data_point)
            if product > max_product:
                max_product = product
                max_class = c
        if max_class != label:
            # update weights
            self.weights[:,label] += data_point
            self.weights[:,max_class] -= data_point
        
        return


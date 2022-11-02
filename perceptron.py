from cmath import inf
import numpy as np

class perceptron:
    def __init__(self, num_features, num_classes):
        self.classes = np.arange(0, num_classes, 1) #range(num_classes) 
        self.weights = np.zeros((num_classes, num_features)) 
        # print (self.weights) # print the weights
        # print (self.classes) # print the classes

    #predict_and_update
    
    def predict(self, data_point):
        # return class label (one of self.classes) based on class data point
        max_product = -inf
        max_class = -1
        for c in self.classes:
            product = np.dot(self.weights[c,:], data_point)
            print ("Estimating the weighted sum for class ", c, ": ", product) # print the product
            if product > max_product:
                max_product = product
                max_class = c
        return max_class

    def fit(self, data_point, label):   #@@@ ADD CONSTANT PARAMETER TO WEIGHTS!!!
        # update weights based on data point and label
        max_product = -inf
        max_class = -1
        for c in self.classes:
            #print ("Data point: ", data_point) # print the data point
            #print ("Weights: ", self.weights[c,:]) # print the weights

            product = np.dot(self.weights[c,:], data_point)
            if product > max_product:
                max_product = product
                max_class = c
        if max_class != label:
            # update weights
            self.weights[label,:] += data_point
            self.weights[max_class,:] -= data_point
#Replace loop with matrix multiplication-based gradient descent
        return


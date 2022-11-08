from cmath import inf
import numpy as np

class perceptron:
    def __init__(self, num_features, num_classes):
        self.classes = np.arange(0, num_classes, 1) 
        self.weights = np.zeros((num_classes, num_features+1))
        # loss should be a sigma function
        self.prob = lambda x: 1/(1+np.exp(-x)) # sigmoid function
        self.initStep = 1e-1
        self.stopTol = 1e-4
        self.stopEpochs = 100

    #def predict_and_update
    
    def predict(self, data_point):
        # return class label (one of self.classes) based on class data point
        max_class = -1
        # Compute max_class as the class with the highest np.dot product between the data point and the weights
        max_class = np.argmax(np.dot(self.weights, data_point))
        return max_class

    def fit(self, data_point, label): 
        M,N = data_point.shape
        epoch = 0
        done = False
        Jnll=[1.0]
        while not done:
            Ji = 0.0
            stepsize, epoch = self.initStep*2.0/(2.0+epoch), epoch+1; # update stepsize
            
            for i in np.random.permutation(M):
                xi, yi = data_point[i], label[i]; # get a random data point
                ri = self.weights@xi.T
                si = np.exp(ri)
                si /= np.sum(si)
                Ji += -np.log(si[yi])
                self.weights[yi,:] += stepsize*xi
                self.weights -= stepsize* si.reshape(-1,1)@xi.reshape(1,-1)

            Javerage = Ji/M
            Jnll.append(Javerage)
            if epoch > self.stopEpochs or np.abs(Jnll[-1] - Jnll[-2]) < self.stopTol:
                done = True
        err = np.sum(np.argmax(self.weights@data_point.T, axis=0) != label)/float(M)
        return err

import tracePredictor
import perceptron
import features
import numpy as np
#import onlineLearning as ol
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.data import DataStream 

history_length = 5
num_of_classes = 4
filePath = 'datafiles/ifthenelse/predictionAccuracy.csv'
#filePath = 'datafiles/libquantum/predictionAccuracy.csv'

### Different feature transformations
print ("Binary classifier transform: ")
classfeatures, classlabels = features.generateTraceFeatures(filePath, history_length, num_of_classes)
print ("Features: ", classfeatures.shape, " Labels: ", classlabels.shape)

print ("Bincode transform: ")
#binfeatures, binlabels = features.generateBincodeTraceFeatures(filePath, history_length, num_of_classes)
#print ("Features: ", binfeatures.shape, " Labels: ", binlabels.shape)

print ("Without transform: ")
#wofeatures, wolabels = features.generateTraceFeaturesWithoutTransform(filePath, history_length, num_of_classes)
#print ("Features: ", wofeatures.shape, " Labels: ", wolabels.shape)



### ONLINE LEARNING
#Without shuffling
features = classfeatures
labels = classlabels

# For debugging purposes, reducing the length of datasets
features = features[:5000]
labels = labels[:5000]

training_dataset = features[:int(len(features)*0.05)]
training_labels = labels[:int(len(labels)*0.05)]
test_dataset = features[int(len(features)*0.05):]
test_labels = labels[int(len(labels)*0.05):]

_, num_of_features = training_dataset.shape
ol_perceptron_model = PerceptronMask()
online_trace_predictor = tracePredictor.onlineTracePredictor(num_of_features, num_of_classes, training_dataset, training_labels, test_dataset, test_labels, ol_perceptron_model) 
ol_error = online_trace_predictor.predict_and_update()
print ("Online learning error: ", ol_error)



### STATIC LEARNING
#Shuffle the data

if False:
    indices = np.arange(binfeatures.shape[0])
    np.random.shuffle(indices)
    features = binfeatures[indices]
    labels = binlabels[indices]

    indices = np.arange(wofeatures.shape[0])
    np.random.shuffle(indices)
    features = wofeatures[indices]
    labels = wolabels[indices]

indices = np.arange(classfeatures.shape[0])
np.random.shuffle(indices)
features = classfeatures[indices]
labels = classlabels[indices]

# For debugging purposes, reducing the length of datasets
features = features[:5000]
labels = labels[:5000]
training_dataset = features[:int(len(features)*0.15)]
training_labels = labels[:int(len(labels)*0.15)]
test_dataset = features[int(len(features)*0.15):]
test_labels = labels[int(len(labels)*0.15):]


perceptron_model = perceptron.perceptron(num_of_features, num_of_classes) # create the perceptron model
trace_predictor = tracePredictor.tracePredictor(num_of_features, num_of_classes, perceptron_model) # create the trace predictor
trace_predictor.training_dataset = training_dataset # set the training data
trace_predictor.training_labels = training_labels # set the training labels
trace_predictor.test_dataset = test_dataset # set the test data
trace_predictor.test_labels = test_labels # set the test labels

print ("Training started!")
trace_predictor.train() # train the model
print ("Training error: ", trace_predictor.training_error) # print the training error
trace_predictor.predict() # test the model
print ("Test error: ", trace_predictor.test_error) # print the test error

# Should call perceptron from here
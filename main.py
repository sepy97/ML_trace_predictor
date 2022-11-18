import tracePredictor
import perceptron
import features
import numpy as np

history_length = 5
num_of_classes = 4
#filePath = 'datafiles/ifthenelse/predictionAccuracy.csv'
filePath = 'datafiles/libquantum/predictionAccuracy.csv'

print ("Binary classifier transform: ")
#classfeatures, classlabels = features.generateTraceFeatures(filePath, history_length, num_of_classes)
#print ("Features: ", classfeatures.shape, " Labels: ", classlabels.shape)

print ("Bincode transform: ")
#binfeatures, binlabels = features.generateBincodeTraceFeatures(filePath, history_length, num_of_classes)
#print ("Features: ", binfeatures.shape, " Labels: ", binlabels.shape)

print ("Without transform: ")
wofeatures, wolabels = features.generateTraceFeaturesWithoutTransform(filePath, history_length, num_of_classes)
print ("Features: ", wofeatures.shape, " Labels: ", wolabels.shape)

#print ("class: ")
#print (classfeatures[:5])
#print ("bin: ")
#print (binfeatures[:5])
print ("wo: ")
print (wofeatures[:5])
#Shuffle the data
if False:
    indices = np.arange(classfeatures.shape[0])
    np.random.shuffle(indices)
    features = classfeatures[indices]
    labels = classlabels[indices]
    indices = np.arange(binfeatures.shape[0])
    np.random.shuffle(indices)
    features = binfeatures[indices]
    labels = binlabels[indices]
indices = np.arange(wofeatures.shape[0])
np.random.shuffle(indices)
features = wofeatures[indices]
labels = wolabels[indices]

# For debugging purposes, reducing the length of datasets
features = features[:5000]
labels = labels[:5000]

training_dataset = features[:int(len(features)*0.05)]
training_labels = labels[:int(len(labels)*0.05)]
test_dataset = features[int(len(features)*0.05):]
test_labels = labels[int(len(labels)*0.05):]
print ("Training dataset: ", training_dataset.shape, " Test dataset: ", test_dataset.shape)
_, num_of_features = training_dataset.shape
print ("Number of features: ", num_of_features)


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
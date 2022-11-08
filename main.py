import tracePredictor
import perceptron
import features
import numpy as np

num_of_features = 5
num_of_classes = 4
filePath = 'datafiles/ifthenelse/predictionAccuracy.csv'
#filePath = 'datafiles/libquantum/predictionAccuracy.csv'

features, labels = features.generateTraceFeatures(filePath, num_of_features)
print ("Features: ", features.shape, " Labels: ", labels.shape)

#Shuffle the data
indices = np.arange(features.shape[0])
np.random.shuffle(indices)
features = features[indices]
labels = labels[indices]

# For debugging purposes, reducing the length of datasets
features = features[:5000]
labels = labels[:5000]

training_dataset = features[:int(len(features)*0.05)]
training_labels = labels[:int(len(labels)*0.05)]
test_dataset = features[int(len(features)*0.05):]
test_labels = labels[int(len(labels)*0.05):]
print ("Training dataset: ", training_dataset.shape, " Test dataset: ", test_dataset.shape)


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
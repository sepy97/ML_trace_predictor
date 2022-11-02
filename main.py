import tracePredictor
import perceptron

num_of_features = 10
num_of_classes = 4
#filePath = 'datafiles/ifthenelse/predictionAccuracy.csv'
filePath = 'datafiles/libquantum/predictionAccuracy.csv'

# Load the training data
print ("Getting training data...")
training_data = []
training_labels = []
lines_to_train = 50
skip_lines = num_of_features  
feature_window = []
with open(filePath, 'r') as f:
    for line in f:
        line = line.split(',')
        line.pop() # remove newline
        feature_window.append(int(line[-1]))
        if skip_lines > 0:
            skip_lines -= 1
            continue
        if lines_to_train == 0:
            break
        feature_window.pop(0) # remove the first element of the feature window
        feature_copy = feature_window.copy() # Have to work with a copy, bcoz feature_window will be changed in the next iteration of the loop
        training_data.append(feature_copy) 
        training_labels.append(int(line[-1]))
        lines_to_train -= 1

# Load the test data
print ("Getting test data...")
test_data = []
test_labels = []
skip_test_lines = lines_to_train
skip_lines = num_of_features 
feature_window = []
lines_to_test = 5000
with open(filePath, 'r') as f:
    for line in f:
        if skip_test_lines > 0:
            skip_test_lines -= 1
            continue
        line = line.split(',')
        line.pop() # remove newline
        feature_window.append(int(line[-1]))
        if skip_lines > 0:
            skip_lines -= 1
            continue
        feature_window.pop(0) # remove the first element of the feature window
        feature_copy = feature_window.copy() # Have to work with a copy, bcoz feature_window will be changed in the next iteration of the loop
        test_data.append(feature_copy) # @@@ Something is wrong here!
        test_labels.append(int(line[-1]))
        lines_to_test -= 1
        if lines_to_test == 0:
            break
'''
print ("Training data: ", training_data) # print the training data
print ("Training labels: ", training_labels) # print the training labels
print ("Test data: ", test_data) # print the test data
print ("Test labels: ", test_labels) # print the test labels
'''

# Create the model
perceptron_model = perceptron.perceptron(num_of_features, num_of_classes) # create the perceptron model
trace_predictor = tracePredictor.tracePredictor(num_of_features, num_of_classes, perceptron_model) # create the trace predictor
trace_predictor.training_dataset = training_data # set the training data
trace_predictor.training_labels = training_labels # set the training labels
trace_predictor.test_dataset = test_data # set the test data
trace_predictor.test_labels = test_labels # set the test labels
print ("Training started!")
trace_predictor.train() # train the model
print ("Training error: ", trace_predictor.training_error) # print the training error
trace_predictor.predict() # test the model
print ("Test error: ", trace_predictor.test_error) # print the test error

# Should call perceptron from here
import numpy as np

def transformFeatures (feature_window, num_of_classes):
    # datapoint: a datapoint from the dataset
    # num_of_classes: number of classes in the dataset
    # returns: a transformed datapoint

    transformed_datapoint = np.array([])
    for i in feature_window:
        transformed_i = np.zeros(num_of_classes) 
        transformed_i[i] = 1
        transformed_datapoint = np.append(transformed_datapoint, transformed_i)
    return transformed_datapoint

def generateTraceFeatures (filePath, num_of_features):
    # filePath: path to the trace data file
    # num_of_features: number of features to be extracted
    # returns: a tuple with feature matrix and label vector

    skip_lines = num_of_features  
    feature_window = []
    features = []
    labels = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.split(',')
            line.pop() # remove newline
            feature_window.append(int(line[-1]))
            if skip_lines > 0:
                skip_lines -= 1
                continue
            feature_window.pop(0) # remove the first element of the feature window
            #print ("Feature window: ", feature_window)
            transformed_feature = transformFeatures(feature_window, 4)
            #print ("Transformed feature: ", transformed_feature)
            #feature_copy = feature_window.copy() # Have to work with a copy, bcoz feature_window will be changed in the next iteration of the loop
            features.append(transformed_feature)#feature_copy)
            labels.append(int(line[-1]))
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)

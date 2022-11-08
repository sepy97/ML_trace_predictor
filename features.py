import numpy as np

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
            feature_copy = feature_window.copy() # Have to work with a copy, bcoz feature_window will be changed in the next iteration of the loop
            features.append(feature_copy)
            labels.append(int(line[-1]))
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)
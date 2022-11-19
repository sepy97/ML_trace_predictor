import numpy as np
import math

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

def generateTraceFeatures (filePath, num_of_features, num_of_classes):
    # filePath: path to the trace data file
    # num_of_features: number of features to be extracted
    # num_of_classes: maximal possible class number in the dataset
    # returns: a tuple with feature matrix and label vector

    skip_lines = num_of_features  
    feature_window = []
    features = []
    labels = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.split(',')
            line.pop() # remove newline
            if skip_lines > 0:
                feature_window.append(int(line[-1]))
                skip_lines -= 1
                continue
            transformed_feature = transformFeatures(feature_window, num_of_classes)
            features.append(transformed_feature)
            labels.append(int(line[-1]))
            feature_window.append(int(line[-1]))
            feature_window.pop(0)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)

def bincodeTransform (feature_window, num_of_classes):
    transformed_datapoint = np.array([])
    for i in feature_window:
        bin_i = format(i, 'b')
        bin_i = bin_i.zfill(int(math.log(num_of_classes,2)))
        transformed_i = np.array(list(bin_i))
        transformed_i = transformed_i.astype(np.float64)
        transformed_datapoint = np.append(transformed_datapoint, transformed_i)
    return transformed_datapoint

def generateBincodeTraceFeatures (filePath, num_of_features, num_of_classes):
    # filePath: path to the trace data file
    # num_of_features: number of features to be extracted
    # num_of_classes: maximal possible class number in the dataset
    # returns: a tuple with feature matrix and label vector

    skip_lines = num_of_features  
    feature_window = []
    features = []
    labels = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.split(',')
            line.pop() # remove newline
            if skip_lines > 0:
                feature_window.append(int(line[-1]))
                skip_lines -= 1
                continue
            transformed_feature = bincodeTransform(feature_window, num_of_classes)
            features.append(transformed_feature)
            labels.append(int(line[-1]))
            feature_window.append(int(line[-1]))
            feature_window.pop(0)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)

def generateTraceFeaturesWithoutTransform (filePath, num_of_features, num_of_classes):
    # filePath: path to the trace data file
    # num_of_features: number of features to be extracted
    # num_of_classes: maximal possible class number in the dataset
    # returns: a tuple with feature matrix and label vector

    skip_lines = num_of_features  
    feature_window = []
    features = []
    labels = []
    with open(filePath, 'r') as f:
        for line in f:
            line = line.split(',')
            line.pop() # remove newline
            if skip_lines > 0:
                feature_window.append(float(line[-1]))
                skip_lines -= 1
                continue
            feature_copy = feature_window.copy()
            features.append(feature_copy)
            labels.append(int(line[-1]))
            feature_window.append(float(line[-1]))
            feature_window.pop(0)
    features = np.array(features)
    labels = np.array(labels)
    return (features, labels)
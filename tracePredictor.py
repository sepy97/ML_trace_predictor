import perceptron
import collections
class tracePredictor:
    def __init__(self, num_features, num_classes, model):
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = model
        self.training_dataset = []
        self.training_labels = []
        self.test_dataset = []
        self.test_labels = []
        self.training_error = 0
        self.test_error = 0

    def predict(self):
        # loop through data in test dataset, call model.predict() on each data point, check if prediction matches actual value, if so, increment correct counter, increment total counter, return correct/total
        correct = 0
        total = 0
        for i in range(len(self.test_dataset)):
            trace_id = self.model.predict(self.test_dataset[i])
            if trace_id == self.test_labels[i]:
                correct += 1
            total += 1
        self.test_error = 1.0 - float(correct)/float(total)
        return self.test_error

    def train(self):
        # loop through data in training dataset, call model.fit() on each data point, return training error
        for i in range(len(self.training_dataset)):
            self.model.fit(self.training_dataset[i], self.training_labels[i])
        total = len(self.training_dataset)
        incorrect = 0
        for i in range(len(self.training_dataset)):
            trace_id = self.model.predict(self.training_dataset[i])
            if trace_id != self.training_labels[i]:
                print ("Incorrect prediction: ", trace_id, " instead of ", self.training_labels[i], " on dataset: ", self.training_dataset[i]) # print the incorrect prediction
                incorrect += 1
        self.training_error = float(incorrect)/float(total)
        return self.training_error

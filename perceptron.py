import numpy as np

class perceptron:

    
    learning_rate = 0.01
    epochs = 100
    input_size = 2
    weights = np.zeros(input_size+1)
    bias = 0

    def __init__(self, input_size, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size+1)
        self.bias = 0

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        a = self.activation_fn(z)
        return a

    def fit(self, X, y):
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                x = np.insert(X[i], 0, 1)
                y_pred = self.predict(x)
                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * x
                self.bias += update
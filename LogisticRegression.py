import numpy as np


class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            equation = np.dot(X, self.weights) + self.bias
            Y_pred = self.sigmoid(equation)

            cost = (-1 / n_samples) * np.sum(Y*np.log(Y_pred) + (1-Y)*np.log(1-Y_pred)) 
            
            dw = (1 / n_samples) * np.dot(X.T, (Y_pred - y))
            db = (1 / n_samples) * np.sum(Y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        equation = np.dot(X, self.weights) + self.bias
        Y_pred = self.sigmoid(equation)
        Y_pred_class = [1 if i > 0.5 else 0 for i in Y_pred]
        return np.array(Y_pred_class)

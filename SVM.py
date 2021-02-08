import numpy as np


class SVM:
    def __init__(self, lr=0.001, lamda=0.01, n_iter=1000):
        self.lr = lr
        self.lamda = lamda
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, Y):
        # First we convert 1 class to -1 if its 0 or someother negative number
        Y = np.where(Y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iter):
            for ind, x in enumerate(X):
                if Y[ind]*(np.dot(x, self.w) - self.b) >= 1:
                    dw = 2 * self.lamda * self.w
                    db = 0
                else:
                    dw = 2 * self.lamda * self.w - np.dot(x, Y[ind])
                    db = Y[ind]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

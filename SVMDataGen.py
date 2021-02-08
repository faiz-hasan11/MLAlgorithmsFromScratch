import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from SVM import SVM

X, Y = datasets.make_blobs(n_features=2, centers=2)

# To plot dataset
# plt.figure(figsize=(8, 8))
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
# plt.show()

Y = np.where(Y <= 0, -1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)


def accuracy(Y, Y_pred):
    acc = np.sum(Y == Y_pred) / len(Y)
    return acc


def best_params():
    lr_list = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    acc_max = 0
    lr_max = 0
    lamda_max = 0
    lambda_list = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    for lr_val in lr_list:
        for lmda in lambda_list:
            clf = SVM(lr=lr_val, lamda=lmda)
            clf.fit(X_train, Y_train)
            predictions = clf.predict(X_test)
            acc = accuracy(Y_test, predictions)
            if acc > acc_max:
                acc_max = acc
                lr_max = lr_val
                lamda_max = lmda
    return (lr_max, lamda_max, acc_max)


lr, lamda, acc = best_params()


print("Best Learning Rate:", lr)
print("Best Lambda Value", lamda)
print("Best Classification Accuracy:", acc)

# Best Learning Rate: 0.1
# Best Lambda Value 0.1
# Best Classification Accuracy: 1.0

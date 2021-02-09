import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from DecisionTree import DecisionTree

dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)


def accuracy(Y, Y_pred):
    acc = np.sum(Y == Y_pred) / len(Y)
    return acc


def best_params():
    acc_max = 0
    depth_max = 0
    depth_list = [i*10 for i in range(1, 21)]
    for depth in depth_list:
        clf = DecisionTree(max_depth=depth)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        acc = accuracy(Y_test, predictions)
        if acc > acc_max:
            acc_max = acc
            depth_max = depth
    return (depth_max, acc_max)


depth, acc = best_params()
print("Best Tree Depth:", depth)
print("Best Classification Accuracy:", acc)

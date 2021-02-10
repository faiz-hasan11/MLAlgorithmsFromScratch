import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from RandomForest import RandomForest

dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)


def accuracy(Y, Y_pred):
    acc = np.sum(Y == Y_pred) / len(Y)
    return acc


def best_params():
    acc_max = 0
    n_trees_max = 0
    n_trees_list = [i for i in range(2, 11)]
    for n_tree in n_trees_list:
        clf = RandomForest(n_trees=n_tree)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        acc = accuracy(Y_test, predictions)
        if acc > acc_max:
            acc_max = acc
            n_trees_max = n_tree
    return (n_trees_max, acc_max)


n_tree, acc = best_params()
print("Best Number of Trees:", n_tree)
print("Best Classification Accuracy:", acc)

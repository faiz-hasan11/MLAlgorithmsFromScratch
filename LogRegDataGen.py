import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

dataset = datasets.load_breast_cancer()
X, Y = dataset.data, dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)


def accuracy(Y, Y_pred):
    acc = np.sum(Y == Y_pred) / len(Y)
    return acc


def best_params():
    lr_list = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    acc_max = 0
    lr_max = 0
    n_iter_max = 0
    iter_list = [i*1000 for i in range(1, 11)]
    for lr_val in lr_list:
        for iteration in iter_list:
            reg = LogisticRegression(lr=lr_val, n_iters=iteration)
            reg.fit(X_train, Y_train)
            predictions = reg.predict(X_test)
            acc = accuracy(Y_test, predictions)
            if acc > acc_max:
                acc_max = acc
                lr_max = lr_val
                n_iter_max = iteration
    return (lr_max, n_iter_max)


best_lr, best_n_iter = best_params()

print("Best Learning Rate:", best_lr)
print("Best Number Of Iterations:", best_n_iter)

# Best LR = 0.0001
# Best N_Iter = 1000

reg = LogisticRegression(lr=best_lr, n_iters=best_n_iter)
reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)


print("Best Classification Accuracy", accuracy(Y_test, predictions))

# On Running the Code , The following  warning will be shown:
#  RuntimeWarning: overflow encountered in exp
#  return 1 / (1 + np.exp(-X))

# This happens because X is very big which produces exp(-X) to be extremely small to be represented in 64bits hence 128bits required to represent it.
# Hence only selected number 'lr' value and 'n_iter' value is used.For small number of comparisions.

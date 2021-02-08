from KNN import KNN
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, Y = iris.data, iris.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)

# Plotting only for 2 features
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='viridis', edgecolors='k', s=20)
# plt.show()


def best_params(acc_max, k_best):
    for k_val in range(3, 11, 2):
        clf = KNN(k=k_val)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        acc = np.sum(predictions == Y_test) / len(Y_test)
        if acc > acc_max:
            acc_max = acc
            k_best = k_val
    return (acc_max, k_best)


acc_max = 0
k_best = 3
acc_max, k_best = best_params(acc_max, k_best)

print("Best Accuracy:", acc_max)
print("Best Value of K:", k_best)

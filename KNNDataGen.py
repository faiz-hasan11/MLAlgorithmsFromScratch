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

clf = KNN(k=3)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == Y_test) / len(Y_test)
print(acc)

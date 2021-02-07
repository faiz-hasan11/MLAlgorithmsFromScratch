import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

X, Y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=7865)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1234)


def mse(Y, Y_pred):
    return np.mean((Y - Y_pred)**2)


def best_params():
    lr_list = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]
    mse_min = 10000000
    lr_min = 0
    n_iter_min = 0
    for lr_val in lr_list:
        for iteration in range(1000, 10000, 10):
            reg = LinearRegression(learning_rate=lr_val, n_iters=iteration)
            reg.fit(X_train, Y_train)
            predicted = reg.predict(X_test)
            mse_val = mse(Y_test, predicted)
            if mse_val < mse_min:
                mse_min = mse_val
                lr_min = lr_val
                n_iter_min = iteration
    return (lr_min, n_iter_min)


best_lr, best_n_iter = best_params()

print("Best Learning Rate:", best_lr)
print("Best Number Of Iterations:", best_n_iter)
# Best LR = 0.001
# Best N_Iter = 2550

reg = LinearRegression(learning_rate=best_lr, n_iters=best_n_iter)
reg.fit(X_train, Y_train)
predictions = reg.predict(X_test)


mse_val = mse(Y_test, predictions)
print("MSE:", mse_val)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, Y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, Y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()

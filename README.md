
MLAlgorithmsFromScratch
======================
This repository contains my understanding and implementation of basic machine learning algorithms from scratch using Numpy. An atempt has been made to find the best hyperparameters to achieve best accuracy.

## Table of content
- [Linear Regression](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/LinearRegression.py)
- [Logistic Regression](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/LogisticRegression.py)
- [KNN](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/KNN.py)

## Algorithms

Simple and Quick explanation of the Algorithms along with Optimization techniques. 

### Linear Regression

Simple linear regression is a type of regression analysis where the number of independent variables is one and there is a linear relationship between the independent(x) and dependent(y) variable.

` y  = w * x + b `

w = weights
b = bias

The motive of the linear regression algorithm is to find the best values for w and b.

#### Cost Function

The cost function helps us to figure out the best possible values for `w` and `b` which would provide the best fit line for the data points. Since we want the best values for `w` and `b`, we convert this search problem into a minimization problem where we would like to minimize the error between the predicted value and the actual value.


![Cost Function](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/1_wQCSNJ486WxL4mZ3FOYtgw.png)

This cost function is also known as the Mean Squared Error(MSE) function. Now, using this MSE function we are going to change the values of `w` and `b` such that the MSE value settles at the minima.

#### Gradient Descent
Gradient descent is a method of updating `w` and `b` to reduce the cost function(MSE). The idea is that we start with some values for `w` and `b` and then we change these values iteratively to reduce the cost. Gradient descent helps us on how to change the values.

![Gradient Descent](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/GradientDescent.png)

To update `w` and `b`, we take gradients from the cost function. To find these gradients, we take partial derivatives with of cost function respect to `w` and `b`. 

![Partial Derivative](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/CostUpdation.png)
Here a0 is b and w1 is w.

#### Learning Rate
Alpha is the learning rate which is a hyperparameter that you must specify. A smaller learning rate could get you closer to the minima but takes more time to reach the minima, a larger learning rate converges sooner but there is a chance that you could overshoot the minima. With the help of Alpha we update `w` and `b`.
 
![Updation](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/update.png)
Here a0 is b and w1 is w.

#### Finding Best HyperParameters
We pass a set of hyperparameter and try to find their best value for best results.
```javascript
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
}
```
#### Dataset

Visulaization of the dataset generated in [Reg Data Gen File](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/RegDataGen.py) for Linear Regression

![Dataset](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/RegDataSet.png)

#### Most Fit Line

Visualization of the most fit line on the given Dataset.

![FitLine](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/OpttimizedRegLine.png)

### Logistic Regression

Logistic Regression is a Machine Learning algorithm which is used for the classification problems, it is a predictive analysis algorithm and based on the concept of probability.
Logistic Regression uses a more complex cost function, this cost function can be defined as the ‘Sigmoid function’.The hypothesis of logistic regression tends it to limit the cost function between 0 and 1.

#### Sigmoid Function

Sigmoid function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.

![Graph](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/sigmoidgraph.png)
![Formula](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/sigmoidformula.png)

#### Hypothesis Representation

When using linear regression we used a formula of the hypothesis i.e.

`y = w * x + b`

For logistic regression we are going to modify it a little bit i.e.

`σ(y) = σ(w * x + b)`

 σ() = Sigmoid Function
 
 ![HypoThesis](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/LogRegHypo.png)
 
#### Decision Boundary

We expect our classifier to give us a set of outputs or classes based on probability when we pass the inputs through a prediction function and returns a probability score between 0 and 1.

#### Cost Function

If we try to use the cost function of the linear regression in ‘Logistic Regression’ then it would be of no use as it would end up being a non-convex function with many local minimums, in which it would be very difficult to minimize the cost value and find the global minimum.

Instead We Use =>

![CostFunction](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/LRCostFunction.png)

#### Gradient Descent

The main goal of Gradient descent is to minimize the cost value. i.e. min J(θ).Now to minimize our cost function we need to run the gradient descent function on each parameter.
Gradient descent has an analogy in which we have to imagine ourselves at the top of a mountain valley and left stranded and blindfolded, our objective is to reach the bottom of the hill. Feeling the slope of the terrain around you is what everyone would do.

![GD](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/GDLR.jpeg)

#### Finding Best HyperParameters

We pass a set of hyperparameter and try to find their best value for best results.

```javascript
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
```
#### Dataset

The Dataset used is the [Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) taken from the SkLearn Library. It has 2 output classes.

#### Accuracy

The model achieved an accuarcy of approx 92% with Learning Rate = 0.0001 and Number Of Iterations = 1000

### KNN

K nearest neighbors is a supervised machine learning algorithm often used in classification problems. It works on the simple assumption that “The apple does not fall far from the tree” meaning similar things are always in close proximity. This algorithm works by classifying the data points based on how the neighbors are classified. Any new case is classified based on a similarity measure of all the available cases.

#### Concepts about KNN

- **Lazy Learning Algorithm** — It is a lazy learner because it does not have a training phase but rather memorizes the training dataset.
- **Case-Based Learning Algorithm** -The algorithm uses raw training instances from the problem domain to make predictions and is often referred to as an instance based or case-based learning algorithm. Case-based learning implies that KNN does not explicitly learn a model. Rather it memorizes the training instances/cases which are then used as “knowledge” for the prediction phase. 
- **Non-Parametric** — A non-parametric method has either a fixed number of parameters regardless of the data size or has no parameters. In KNN, irrespective of the size of data, the only unknown parameter is K.

#### What is K in KNN algorithm?

K in KNN is the number of nearest neighbors considered for assigning a label to the current point. K is an extremely important parameter and choosing the value of K is the most critical problem when working with the KNN algorithm. The process of choosing the right value of K is referred to as parameter tuning and is of great significance in achieving better accuracy. Most data scientists usually choose an odd number value for K when the number of classes is 2.

#### When should you use KNN Algorithm?

KNN algorithm is a good choice if you have a small dataset and the data is noise free and labeled. When the data set is small, the classifier completes execution in shorter time duration. If your dataset is large, then KNN, without any hacks, is of no use.

#### How does KNN work?

- Choose a value for K. K should be an odd number.
- Find the distance of the new point to each of the training data.
- Find the K nearest neighbors to the new data point.
- For classification, count the number of data points in each category among the k neighbors. New data point will belong to class that has the most neighbors.

#### How is the distance calculated?

Distance can be calculated using =>
- Euclidean distance
- Manhattan distance
- Hamming Distance
- Minkowski Distance

Here I have used Euclidean Distance.

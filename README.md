
MLAlgorithmsFromScratch
======================
This repository contains my understanding and implementation of basic machine learning algorithms from scratch using Numpy. An atempt has been made to find the best hyperparameters to achieve best accuracy.

## Table of content
- [Linear Regression](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/LinearRegression.py)
- [Logistic Regression](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/LogisticRegression.py)
- [K Nearest Neighbours](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/KNN.py)
- [Support Vector Machines](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/SVM.py)
- [Decision Trees](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/DecisionTree.py)
- [Random Forest](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/RandomForest.py)

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
![Euclidean](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/euclidean.png)

#### Finding Best HyperParameters

We pass a set of hyperparameter and try to find their best value for best results.

```javascript
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
```

#### Dataset

The Dataset used is the [Iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) taken from the SkLearn Library. It has 3 output classes.

#### Accuracy

The model achieved an accuracy of approx 96.7% with k = 5

### Support Vector Machines

Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression tasks. However, it is mostly used in classification problems.We plot each data item as a point in n-dimensional space (where n is number of features) with the value of each feature being the value of a particular coordinate. Then, we perform classification by finding the hyper-plane that differentiates the two classes very well.

#### Hyperplanes

Hyperplanes are decision boundaries that help classify the data points. Data points falling on either side of the hyperplane can be attributed to different classes. Also, the dimension of the hyperplane depends upon the number of features

![HyperPlane](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/hyperplane.png)

Equation of the HyperPlane is =>

![Equation](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/SVMEquation.png)

#### Hinge Loss

In the SVM algorithm, we want to maximize the margin between the data points and the hyperplane. So we do it with the help of Hinge Loss function.If we’re on the right side of the graph, which means that the predicted and actual value have the same sign, then our cost becomes zero. Otherwise, we have a particular loss value.

![HingeLoss](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/HingeLoss.png)

#### Regularization

We also add a regularization parameter the cost function. The objective of the regularization parameter is to balance the margin maximization and loss. After adding the regularization parameter, the cost functions looks as below.

![Reqularization](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/SVMregularization.png)

#### Gradient Descent 

To minimize the loss , we use Gradient Descent.

![GD](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/SVMGD.png)

#### Finding Best HyperParameters

We pass a set of hyperparameter and try to find their best value for best results.

```javascript
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
```

#### Dataset

Visulaization of the dataset generated in [SVM Data Gen File](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/SVMDataGen.py) for SVM classification.

![Dataset](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/SVMdata.png)


#### Accuracy

The model achieved an accuracy of 100% with Learning Rate =  0.1 and Lambda Value =  0.1

### Decision Trees

A decision tree is one of the supervised machine learning algorithms, this algorithm can be used for regression and classification problems. It is mostly used for classification problems. A decision tree follows a set of if-else conditions to visualize the data and classify it according to the conditions.

![DecisionTree](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/DecisionTree.png)

#### Working of Decision Tree

The root node feature is selected based on the results from the Attribute Selection Measure(ASM).The ASM is repeated until there is a leaf node or a terminal node where it cannot be split into sub-nodes.

![Working](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/DTWorking.png)

#### Attribute Selective Measure(ASM)

Attribute Subset Selection Measure is a technique used in the data mining process for data reduction. The data reduction is necessary to make better analysis and prediction of the target variable. The two main ASM techniques are =>
- Gini index
- Information Gain(ID3)

Here I have used Information Gain.

#### Information Gain

Entropy is the main concept of this algorithm which helps in determining a feature or attribute that gives maximum information about a class is called Information gain or ID3 algorithm. By using this method we can reduce the level of entropy from the root node to the leaf node.

![Entropy](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/entropy.png)

‘p’, denotes the probability of E(S) which denotes the entropy. The feature or attribute with the highest Information gain is used as the root for the splitting.

#### Finding Best HyperParameters

We pass a set of hyperparameter and try to find their best value for best results.

```javascript
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
```

#### Dataset

The Dataset used is the [Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) taken from the SkLearn Library. It has 2 output classes.

#### Accuracy

The model achieved an accuracy of approx 93% with Max Depth =  50

### Random Forest

The Random Forest Algorithm is composed of different decision trees, each with the same nodes, but using different data that leads to different leaves. It merges the decisions of multiple decision trees in order to find an answer, which represents the average of all these decision trees.

![RF](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/RandomForest.png)

#### Bootstrapping

When training, each tree in a random forest learns from a random sample of the data points. The samples are drawn with replacement, known as bootstrapping, which means that some samples will be used multiple times in a single tree. The idea is that by training each tree on different samples, although each tree might have high variance with respect to a particular set of the training data, overall, the entire forest will have lower variance but not at the cost of increasing the bias.At test time, predictions are made by averaging the predictions of each decision tree.

![BootStrap](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/Bootstrapping.png)

#### Finding Best HyperParameters

We pass a set of hyperparameter and try to find their best value for best results.

```javascript
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
```

#### Dataset

The Dataset used is the [Breast Cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) taken from the SkLearn Library. It has 2 output classes.

#### Accuracy

The model achieved an accuracy of approx 95% with Number Of Trees = 6

##### Made by Syed Faiz Hasan :wave:

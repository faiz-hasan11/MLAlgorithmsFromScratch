
MLAlgorithmsFromScratch
======================
This repository contains my understanding and implementation of basic machine learning algorithms from scratch using Numpy..

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
Here a0 is w and w1 is b.

#### Learning Rate
Alpha is the learning rate which is a hyperparameter that you must specify. A smaller learning rate could get you closer to the minima but takes more time to reach the minima, a larger learning rate converges sooner but there is a chance that you could overshoot the minima. With the help of Alpha we update `w` and `b`.
 
![Updation](https://github.com/faiz-hasan11/MLAlgorithmsFromScratch/blob/master/Images/update.png)
Here a0 is w and w1 is b.

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
### Database setup

If you use MySQL < 5.7.8, you have to use `utf8` and `utf8_unicode_ci` instead because those MySQL versions can't handle the long indexes created by `utf8mb4` (up to four bytes per character) and you will get errors like

```
1071 Specified key was too long; max key length is 767 bytes
```

To avoid that, change your database settings in your `./typo3conf/LocalConfiguration.php` to:

```
'DB' => [
    'Connections' => [
        'Default' => [
            'tableoptions' => [
                'charset' => 'utf8',
                'collate' => 'utf8_unicode_ci',
            ],
            // ...
        ],
    ],
],
```

### Security

Since **TYPO3 9.5.14+** implements **SameSite cookie handling** and restricts when browsers send cookies to your site. This is a problem when customers are redirected from external payment provider domain. Then, there's no session available on the confirmation page. To circumvent that problem, you need to set the configuration option `cookieSameSite` to `none` in your `./typo3conf/LocalConfiguration.php`:

```
    'FE' => [
        'cookieSameSite' => 'none'
    ]
```

### Extension

* Log into the TYPO3 back end
* Click on ''Admin Tools::Extension Manager'' in the left navigation
* Click the icon with the little plus sign left from the Aimeos list entry (looks like a lego brick)

![Install Aimeos TYPO3 extension](https://aimeos.org/docs/images/Aimeos-typo3-extmngr-install.png)

### Database

Afterwards, you have to execute the update script of the extension to create the required database structure:

![Execute update script](https://aimeos.org/docs/images/Aimeos-typo3-extmngr-update-7.x.png)

## Page setup

The page setup for an Aimeos web shop is easy if you import the example page tree for TYPO3 9/10:

* [20.10.x page tree](https://aimeos.org/fileadmin/download/Aimeos-pages_20.10.t3d)
* [19.10.x page tree](https://aimeos.org/fileadmin/download/Aimeos-pages_two-columns_18.10__2.t3d)

**Note:** The Aimeos layout expects [Bootstrap](https://getbootstrap.com) providing the grid layout!

### Go to the import view

* In Web::Page, root page (the one with the globe)
* Right click on the globe
* Move the cursor to "Branch actions"
* In the sub-menu, click on "Import from .t3d"

![Go to the import view](https://aimeos.org/docs/images/Aimeos-typo3-pages-menu.png)

### Upload the page tree file

* In the page import dialog
* Select the "Upload" tab (2nd one)
* Click on the "Select" dialog
* Choose the file you've downloaded
* Press the "Upload files" button

![Upload the page tree file](https://aimeos.org/docs/images/Aimeos-typo3-pages-upload.png)

### Import the page tree

* In Import / Export view
* Select the uploaded file from the drop-down menu
* Click on the "Preview" button
* The pages that will be imported are shown below
* Click on the "Import" button that has appeared
* Confirm to import the pages

![Import the uploaded page tree file](https://aimeos.org/docs/images/Aimeos-typo3-pages-import.png)

Now you have a new page "Shop" in your page tree including all required sub-pages.

### SEO-friendly URLs

TYPO3 9.5 and later can create SEO friendly URLs if you add the rules to the site config:
[https://aimeos.org/docs/latest/typo3/setup/#seo-urls](https://aimeos.org/docs/latest/typo3/setup/#seo-urls)

## License

The Aimeos TYPO3 extension is licensed under the terms of the GPL Open Source
license and is available for free.

## Links

* [Web site](https://aimeos.org/integrations/typo3-shop-extension/)
* [Documentation](https://aimeos.org/docs/TYPO3)
* [Forum](https://aimeos.org/help/typo3-extension-f16/)
* [Issue tracker](https://github.com/aimeos/aimeos-typo3/issues)
* [Source code](https://github.com/aimeos/aimeos-typo3)

import numpy as np
from DecisionTree import DecisionTree
from collections import Counter

# This method is used to generate subsamples of data for different trees.


def BootstrapSample(X, Y):
    n_samples = X.shape[0]
    ind = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[ind], Y[ind]

# This method returns the most common label


def MostCommonLabel(Y):
    counter = Counter(Y)
    mostcommon = counter.most_common(1)[0][0]
    return mostcommon


class RandomForest:

    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, Y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, Y_sample = BootstrapSample(X, Y)
            tree.fit(X_sample, Y_sample)
            self.trees.append(tree)

    # In this method we find the most common label generated from different Trees.
    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(preds, 0, 1)
        Y_pred = [MostCommonLabel(pred) for pred in preds]
        return np.array(Y_pred)

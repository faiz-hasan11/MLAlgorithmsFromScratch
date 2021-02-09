import numpy as np
from collections import Counter

# Function to calculate Entropy


def entropy(Y):
    hist = np.bincount(Y)  # Number of occurences of all class labels
    ps = hist / len(Y)
    return -np.sum([p*np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # Function to check if its a leaf node.If its a leaf node it has "value"
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, Y):
        self.n_feats = X.shape[1]
        self.root = self.GrowTree(X, Y)

    # GrowTree Function recursively generates the Tree
    def GrowTree(self, X, Y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(Y))

        # Terminating condition => If the tree has attained max depth or node has only 1 type of labels or No. of samples in a node are less than min_samples_split
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self.MostCommonLabel(Y)
            # Return the node with "value"."Value" denotes the label that node has.
            return Node(value=leaf_value)

        # First for the node randomly select a feature or subset of features
        feature_ind = np.random.choice(n_features, self.n_feats, replace=False)

        # Then for the selected features,we select the best feature along with their thresholds
        best_feat, best_thresh = self.BestCriteria(X, Y, feature_ind)

        # We split the node on the basis of best feature and its best threshold into left and right child nodes
        left_ind, right_ind = self.Split(X[:, best_feat], best_thresh)
        left = self.GrowTree(X[left_ind, :], Y[left_ind], depth+1)
        right = self.GrowTree(X[right_ind, :], Y[right_ind], depth+1)
        # Return the best feature along with its best threshold for that node required to split into its children
        return Node(best_feat, best_thresh, left, right)

    # Calculates the best feature and the best threshold greedily by going through all features and all its unique values as threshold and calualating InfoGain
    def BestCriteria(self, X, Y, feature_ind):
        best_gain = -1
        split_ind, split_thresh = None, None
        for ind in feature_ind:
            X_col = X[:, ind]
            thresholds = np.unique(X_col)
            for thresh in thresholds:
                gain = self.InfoGain(Y, X_col, thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_ind = ind
                    split_thresh = thresh

        return split_ind, split_thresh

    # Calculates Info Gain for the current node with given feature and threshold
    def InfoGain(self, Y, X_col, split_thresh):
        parent_entropy = entropy(Y)
        left_ind, right_ind = self.Split(X_col, split_thresh)

        if len(left_ind) == 0 or len(right_ind) == 0:
            return 0

        l_entropy = entropy(Y[left_ind])
        r_entropy = entropy(Y[right_ind])

        child_entropy = (len(left_ind) / len(Y)) * l_entropy + \
            (len(right_ind)/len(Y)) * r_entropy

        ig = parent_entropy - child_entropy
        return ig

    # Splits the tree into left and right child based on best feature and best threshold value
    def Split(self, X_col, split_thresh):
        left_ind = np.argwhere(X_col <= split_thresh).flatten()
        right_ind = np.argwhere(X_col > split_thresh).flatten()
        return left_ind, right_ind

    # Returns the mose common label in the leaf node to be stored as 'Value'
    def MostCommonLabel(self, Y):
        counter = Counter(Y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    # Traverse the whole tree to predict the value.
    def predict(self, X):
        return np.array([self.TraverseTree(x, self.root) for x in X])

    def TraverseTree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.TraverseTree(x, node.left)
        return self.TraverseTree(x, node.right)

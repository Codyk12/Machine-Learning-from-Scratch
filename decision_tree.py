import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from numpy import log2

class DTNode:

    def __init__(self, data, splits, prev_splits=[]):
        """

        :param split_data: data split into n
        """
        self.data = data
        self.splits = splits
        self.split_idx = None
        self.children = {}
        self.prev_splits = prev_splits
        self.most_common_split = None

        return

    def depth(self):
        """
        gets depth of the tree
        :return:
        """
        depths = [0]

        for child in self.children:
            if type(child) == DTNode:
                depths.append(child.depth())
            elif type(child) == DTLeafNode:
                return depths.append(1)

        return 1 + max(depths)

    def predict(self, x):
        """
        Recurse down the tree to the bottom to get label
        :param x: input to classify
        :return:
        """
        if self.split_idx == None:
            return self.children["label"].predict(x)

        f_idx = x[self.split_idx]
        if f_idx in self.children.keys():
            return self.children[f_idx].predict(x)
        else:
            return self.get_most_common_split().predict(x)

    def get_most_common_split(self):
        """
        gets the most common split given data feature not in splits
        :return:
        """
        return self.children[self.most_common_split]


class DTLeafNode(DTNode):

    def __init__(self, label):
        super().__init__(None, None)
        self.label = label

    def predict(self, x):
        return self.label

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            counts = how many types for each attribute
        Example:
            DT  = DTClassifier()
        """
        self.counts = counts
        self.root = None

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        X = np.hstack((X,y))

        # initialize root
        self.root = DTNode(X, self.get_splits(X, []))

        self.recurse_learn(self.root)

        return self

    def recurse_learn(self, node):
        """
        Recurse on the given node and expand children
        :param node:
        :return:
        """
        if len(np.unique(node.data[:,-1])) == 1:
            # add a lead node with the correct label
            self.add_children(node,"label")
            return

        else:
            # add children
            self.split_node(node)
            for k in node.children.keys():
                # recurse
                self.recurse_learn(node.children[k])

        return

    def split_node(self, node):
        """
            Finds the best feature to split on the given node
        :param node: node to split on
        :return:
        """
        # get minimum entropy feature
        gain_idx = self.find_max_gain(node)

        # add the children
        self.add_children(node, gain_idx)

        return

    def find_max_gain(self, node):
        """
        Finds the feature with the highest information gain from the splits in the given node
        :param DTNode: node
        :return:
        """
        infos = []
        keys = list(node.splits.keys())
        for k in keys:
            split = node.splits[k]
            total = split['total']
            del split['total']

            info = 0
            for k in split.keys():
                d = split[k]
                n = len(d)
                # calcualte entropy for the given feature based on labels
                s = self.split_data(d, len(d[0])-1)
                info += n/total*sum([(-len(s)/n)*log2(len(s)/n)])

            infos.append(info)

        # get smallest entropy feature
        max_idx = keys[np.argmin(infos)]

        # keep track of features used
        node.split_idx = max_idx
        return max_idx

    def add_children(self, node, split_idx):
        """
        Adds children to the node
        :param node:
        :param split_idx:
        :return:
        """
        if split_idx == "label":
            node.children['label'] = DTLeafNode(node.data[0, -1])
            return self

        # save children as that split
        split = node.splits[split_idx]
        max_data = 0
        max_data_idx = 0
        for k in split.keys():
            data = split[k]
            splits = self.get_splits(data, node.prev_splits)
            node.children[k] = DTNode(data, splits, node.prev_splits + [split_idx])
            if len(data) > max_data:
                max_data = len(data)
                max_data_idx = k

        node.most_common_split = max_data_idx
        return self


    def get_splits(self, data, prev):
        """
        Gets splits along all features in the data
        :param data:
        :return:
        """
        splits = {}
        for i in range(len(data[0]) - 1):
            if i not in prev:
                splits[i] = self.split_data(data, i)

        return splits

    def split_data(self, data, i):
        """
        gets splits data for a given feature

        :param data: data to split
        :param i: feature number to split data on
        :return: dict: all splits based on unique labels
        """
        feature = data[:,i]
        vals = np.unique(feature)
        splits = {}

        t = 0
        for val in vals:
            s = data[feature == val]
            t += len(s)
            splits[val] = s

        # include total data points across splits
        splits['total'] = t
        return splits

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        return np.array([self.root.predict(x) for x in X])


    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-li    def _shuffle_data(self, X, y):
        """
        s = self.predict(X) == y.flatten()
        return sum(s) / len(s)
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from toolkit import baseline_learner, manager, arff, graph_tools
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=False, Deterministic=10, sc=.01):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.epochs = Deterministic
        self.lr = lr
        self.shuffle = shuffle
        self.sc = sc
        self.miss_classify = []
        self.stopped_epoch = 0

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """

        X = np.hstack((X, np.array([np.ones(len(X))]).T))
        y = y.flatten()
        n = len(y)

        self.weights = self.initialize_weights(len(X[0])) if not initial_weights else initial_weights

        for j in range(self.epochs):

            prev_weights = self.weights.copy()
            miss_classify = 0

            for i in range(n):
                x = X[i,:]

                # get output from weights
                z = 1 if self.weights@x > 0 else 0

                # get weight change
                dw = self.lr*(y[i] - z)*x

                if y[i] != z:
                    miss_classify += 1

                # update weight
                self.weights += dw

            self.miss_classify.append(miss_classify/n)

            # check is norm of weights is above threshold
            if np.mean(abs(self.weights - prev_weights)) < self.sc:
                self.stopped_epoch = j
                return self

            if self.shuffle:
                X, y = self._shuffle_data(X,y)

        return self

    def predict(self, X, s=False):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        if len(X[0]) != len(self.weights):
            X = np.hstack((X, np.array([np.ones(len(X))]).T))
        results = X @ self.weights
        scores = results.copy()
        results[results <= 0] = 0
        results[results > 0] = 1

        if s:
            return results, scores
        else:
            return results

    def initialize_weights(self,n):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        return np.array([0.]*n)

    def score(self, X, y, s=False):
        """
        Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        predictions = self.predict(X, s)
        y = y.flatten()
        return sum(predictions == y) / len(y)

    def _shuffle_data(self, X, y):
        """
        Shuffle the data!
        """
        perm = np.random.permutation(range(len(y)))
        return X[perm], y[perm]

    def get_weights(self):
        return self.weights
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from numpy import outer

def mse(x,y):
    return np.mean((x-y)**2)

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, hidden_layer_widths,
                 activation=lambda x: 1 / (1 + np.exp(-x)), dactivation = lambda x: x*(1-x),
                 lr=.1, momentum=0., w_init=np.random.random, shuffle=True, epochs=1, val_window=10):
        """ Initialize class with chosen hyperparameters.

        Args:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.window = val_window
        self.epochs = epochs
        self.hidden_layer_widths = np.array(hidden_layer_widths)
        self.lr = lr
        self.mom = momentum
        self.w_init = w_init
        self.shuffle = shuffle
        self.layers = None
        self.dw = None
        self.activation = activation
        self.d_activation = dactivation
        self.stopped_epoch = 0

        self.train_mse = []
        self.val_mse = []
        self.acc = []

    def fit(self, X, y, val=None, initial_weights=None, nominal=True):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        y = y.flatten() if nominal else y
        num_out = len(np.unique(y)) if nominal else len(y[0])

        # split validation set
        if val is not None:
            train_X, train_y, val_X, val_y = train_test_split(X, y, test=val)

        num_dp = len(train_y)

        # initialize weights and layers
        self.layers = self.initialize_weights(len(X[0])+1, num_out) if initial_weights is None else initial_weights
        self.dw = self.layers*0
        num_out = len(self.layers[-1])

        for j in range(1,self.epochs+1):

            self.prev_layers = self.layers.copy()
            miss_classify = 0

            # loop over each data point in the dataset
            for i in range(num_dp):
                # one hot for actual class label
                x = train_X[i,:]

                # forward pass through the network
                self.forward(x)

                # one hot encode if nominal data
                if nominal and num_out > 1:
                    y_ = np.zeros(num_out)
                    y_[int(train_y[i])] = 1
                else:
                    y_ = train_y[i]

                # calculate gradient weight change (back prop)
                self.backward(y_)

                # update weights calculated
                self.update_weights()

            # collect the proper metrics
            if val is not None:
                self.collect_metrics(train_X, train_y, val_X, val_y)

                # check is norm of weights is above the stopping criteria threshold
                if self.stop_training():
                    self.stopped_epoch = j
                    return self

            if self.shuffle:
                X, y = self._shuffle_data(X, y)

        self.stopped_epoch = j
        return self

    def initialize_weights(self, n, out):
        """
        Initialize weights for perceptron

        Returns: ndarray of weight matricies

        """
        init = self.w_init
        hlw = self.hidden_layer_widths

        # first layer to hidden layer
        layers = [init((hlw[0], n))]

        # hidden layers
        for i in range(1, len(hlw)):
            layers.append((init((hlw[i],hlw[i-1]+1))))

        # final layer
        layers.append(init((out,hlw[-1]+1)))

        return np.array(layers)

    def forward(self, x_):
        """
        Passes the data forward through the network

        :param x: input value
        :return: output of network
        """
        x = np.concatenate((x_.flatten(), [1]))
        outs = [x]

        for i,layer in enumerate(self.layers):
            outs.append(self.activation(layer@outs[-1]))
            if i < len(self.layers)-1:
                outs[-1] = np.concatenate((outs[-1], [1]))

        self.outs = np.array(outs)
        return outs[-1]


    def backward(self, y):
        """
        calculates gradient backwards through the net
        :param z: output of the network
        :return:
        """
        layers = self.layers
        outs = self.outs

        # delta for output layer
        delta = np.array((y - outs[-1])*self.d_activation(outs[-1]))
        for i in range(len(layers))[::-1]:
            # calculate weight changes
            self.dw[i] = self.lr * outer(delta, outs[i].T) + (self.mom * self.dw[i])

            # update deltas for current layer, but remove bias term
            if i > 0:
                delta = ((layers[i].T*delta).T*self.d_activation(outs[i])).sum(axis=0)[:-1]
        return

    def update_weights(self):
        """
        updates the weights calculated by backprop
        :return:
        """
        for i, dw in enumerate(self.dw):
            self.layers[i] += dw
        return

    def stop_training(self):
        """
        Determines if the accuracy is changing enough based on the validation set
        :return: bool, stop updating
        """

        if len(self.acc) > self.window and len(set(self.acc[-self.window:])) == 1:
            return True
        else:
            return False

    def collect_metrics(self, X, y, val_X, val_y):
        """
        Collects the proper metrics
        :return:
        """
        self.train_mse.append(mse(self.predict(X), y))
        self.val_mse.append(mse(self.predict(val_X), val_y))
        self.acc.append(self.score(val_X, val_y))
        return

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        return np.array([np.argmax(self.forward(X[i,:])) for i in range(len(X))])

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X).reshape(-1)
        y = y.flatten()
        return sum(predictions == y) / len(y)

    def _shuffle_data(self, X, y):
        """
        Shuffle the data
        """
        perm = np.random.permutation(range(len(y)))
        return X[perm], y[perm]

    def get_weights(self):
        return self.layers
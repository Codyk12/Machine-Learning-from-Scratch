import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from numpy import outer

def mse(x,y):
    """
        mean square error
    """
    return np.mean((x-y)**2)

def sigmoid(x):
    """
        sigmoid function
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_dx(x):
    """
        derivative of sigmoid function
    """
    return x*(1-x)

def difference(x_, y):
    """
        Calculates the difference between the target label and the network output

        params:
            x_ (ndarray): network outputs
            y (ndarray): target label
        return:
            (ndarray) subracted difference
    """
    return np.array((y - x_))

class MLPClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, hidden_layer_widths,
                 activation=sigmoid, dactivation=sigmoid_dx, loss_fn = difference,
                 lr=.1, momentum=0., w_init=np.random.random, shuffle=True, epochs=1, val_window=10):
        """ Initialize class with chosen hyperparameters.

        params:
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer
            activation: Non-linear activation function - sigmoid
            dactivation: precalculated derivative of activation function
            lr (float): A learning rate / step size.
            momentum: decay of the step size
            shuffle: Whether to shuffle the training data each epoch.
            epochs: how many training runs to do through the data
            w_init: function to set inital weights for the network
            val_window: Number of training accuracies to keep track of, training will stop when the accuracies dont change for the length of the window

        Example:
            mlp = MLPClassifier([3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.window = val_window
        self.epochs = epochs
        self.hidden_layer_widths = np.array(hidden_layer_widths) # ex. [[1],[2],[3]]
        self.lr = lr
        self.mom = momentum
        self.w_init = w_init # ex. np.random.random
        self.shuffle = shuffle
        self.layers = None
        self.dw = None # matrix for weight (changes) deltas
        self.activation = activation
        self.d_activation = dactivation
        self.loss_fn = loss_fn
        self.stopped_epoch = 0

        self.train_mse = []
        self.val_mse = []
        self.acc = []

        super.__init__()

    def fit(self, X, y, val_split=.25, initial_weights=None, nominal=True):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        params:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            test_split: (double): percent to split validation set
            initial_weights (array-like): allows the user to provide initial weights
            nominal (bool): is data nomial data

        return:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # create the output layer to output 1 number for each class in the data if nomial
        y = y.flatten() if nominal else y
        num_out = len(np.unique(y)) if nominal else len(y[0])

        # split into train/validation data sets
        train_X, train_y, val_X, val_y = train_test_split(X, y, test=val_split)

        # get number of data points
        num_dp = len(train_y)

        # initialize weights and layers
        self.initialize_weights(initial_weights, len(X[0])+1, num_out)

        # train for number of given epochs
        for j in range(self.epochs):
            self.stopped_epoch = j+1

            # loop over each data point in the training data
            for i in range(num_dp):

                # get the training data
                x = train_X[i,:]

                # forward pass through the network
                self.forward(x)

                # get label value
                if nominal and num_out > 1:
                    # one hot encode if nominal data
                    y_ = np.zeros(num_out)
                    y_[int(train_y[i])] = 1
                else:
                    y_ = train_y[i]

                # calculate gradient weight change (back prop)
                self.backward(y_)

                # update weights calculated
                self.update_weights()

            # collect the proper metrics
            if val_split is not None:
                self.collect_metrics(train_X, train_y, val_X, val_y)

            # check is norm of weights is above the stopping criteria threshold
            if self.stop_training():
                return self

            if self.shuffle:
                train_X, train_y = self._shuffle_data(train_X, train_y)

        return self

    def initialize_weights(self, initial_weights, num_in, num_out):
        """
        Initialize the weights for the layers of the network

        params:
            num_in: number of inputs to first layer
            num_out: number of outputs for last layer

        return:
            ndarray of weight matricies
        """
        if initial_weights is not None:
             self.layers = initial_weights
        else:
            init = self.w_init
            hlw = self.hidden_layer_widths

            # first layer to hidden layer
            layers = [init((hlw[0], num_in))]

            # hidden layers
            for i in range(1, len(hlw)):
                layer = init((hlw[i],hlw[i-1]+1)) # +1 for the bias term
                layers.append(layer)

            # final layer
            layers.append(init((num_out,hlw[-1]+1)))

            self.layers = np.array(layers)

        # deltas place holder for calculated weight updates
        self.dw = self.layers*0


    def forward(self, x_):
        """
        Passes the data forward through the network
        saving output at each layer

        params:
            x_: input value
        """
        x = np.concatenate((x_.flatten(), [1])) #add 1 for the bias
        outs = [x]

        for i,layer in enumerate(self.layers):
            outs.append(self.activation(layer@outs[-1])) # outer vector multiply
            if i < len(self.layers)-1:
                outs[-1] = np.concatenate((outs[-1], [1])) # if not last layer tag bias back on

        self.outs = np.array(outs)


    def backward(self, y):
        """
        calculates and saves the gradient calculated from the
        loss propagating backwards through the layers

        params:
            y: target label
        """
        layers = self.layers
        outs = self.outs

        # calculate given loss
        loss = self.loss_fn(y, outs[-1])

        # delta for output layer backprop
        # output loss * derivative of the output
        delta = loss*self.d_activation(outs[-1])

        for i in range(len(layers))[::-1]: # work backwards
            # calculate weight changes
            # multiply the delta calculated for last layer by the pervious layers output by the learning rate
            # add the momentum term based on size of previous step
            self.dw[i] = self.lr * outer(delta, outs[i].T) + self.mom * self.dw[i]

            # while not the first layer update deltas(backprop) for previous layer, but without the bias term
            if i > 0:
                delta = ((layers[i].T*delta).T*self.d_activation(outs[i])).sum(axis=0)[:-1]

    def update_weights(self):
        """
        updates the weights calculated by backprop
        """
        for i, dw in enumerate(self.dw):
            self.layers[i] += dw

    def stop_training(self):
        """
        Determines if the accuracy still changing based on validation window
        If accuracy stops changing within the last number of val_window epochs
        then training should stop

        return:
            (bool) stop training
        """
        return len(self.acc) > self.window and len(set(self.acc[-self.window:])) == 1

    def collect_metrics(self, train_X, train_y, val_X, val_y):
        """
        Collects the proper metrics

        params:
            train_X (ndarray): A 2D numpy array with the training data
            train_y (ndarray): A 2D numpy array with the training labels
            val_X (ndarry): A 2D numpy array with the validation data
            val_y (ndarry): A 2D numpy array with the validation labels
        """
        self.train_mse.append(mse(self.predict(train_X), train_y))
        self.val_mse.append(mse(self.predict(val_X), val_y))
        self.acc.append(self.score(val_X, val_y))

    def predict(self, X):
        """ Predict all classes for a dataset X

        params:
            X (ndarray): A 2D numpy array with the training data, excluding targets

        return:
            array, shape (n_samples): Predicted target values per element in X.
        """
        return np.array([np.argmax(self.forward(X[i,:])) for i in range(len(X))])

    def score(self, X, y):
        """ Return accuracy of model on a given dataset.

        params:
            X (ndarray): A 2D numpy array with data, excluding targets
            y (ndarray): A 2D numpy array with targets

        return:
            score (float): Mean accuracy of self.predict(X) wrt. y.
        """
        predictions = self.predict(X).reshape(-1)
        y = y.flatten()
        return sum(predictions == y) / len(y)

    def _shuffle_data(self, X, y):
        """
        Shuffle the data

        params:
            X (ndarray): data
            y (ndarray): labels
        """
        perm = np.random.permutation(range(len(y)))
        return X[perm], y[perm]

    def get_weights(self):
        """
            gets weights of the class

            return:
                (ndarray) weight layers for the class
        """
        return self.layers

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats


class KNNClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,label_type="class",weight_type='no_weight', k=3, c_mask=None, n_mask=None): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal.
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.label_type = label_type
        self.weight_type = weight_type
        self.k = k
        self.data = None
        self.labels = None
        self.norm = "mixed" if c_mask is not None else "2"
        self.c_mask = c_mask
        self.n_mask = n_mask


    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.X = data
        self.y = labels

        return self

    def predict(self,X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        if self.label_type == "class":
            return np.array([self._pred_class(x) for x in X])
        else:
            return np.array([self._pred_regress(x) for x in X])


    def _get_distances(self, x):
        """
        Gets the distance based on self.norm
        :param x:
        :return:
        """
        # sorted points by distance
        if self.norm == "2":
            return np.linalg.norm(self.X - x, axis=1)
        elif self.norm == "mixed":
            # get continuous distances
            cont_dist = np.linalg.norm(self.X[:, self.c_mask] - x[self.c_mask], axis=1)
            # get nominal distances
            nominal_dist = np.sum(self.X[:,self.n_mask] == x[self.n_mask], axis=1)
            # get missing value distances
            unknown_dist = np.sum(~(np.isnan(self.X) & np.isnan(x)), axis=1)
            return cont_dist + nominal_dist + unknown_dist

    def _get_closest(self, x, k):
        """

        :param x: new data point to classify
        :return:

        """
        distances = self._get_distances(x)

        srtd_idx = np.argsort(distances)

        # get sorted distances
        sorted_dist = distances[srtd_idx][:k]

        # get inidices of top k closest data points
        pnts = srtd_idx[:k]

        # get the k nearest labels
        nearest_labels = self.y[pnts]

        return nearest_labels, sorted_dist

    def _pred_class(self, x):
        """
        Prediction algorithm for classification
        :param data:
        :param labels:
        :return:
        """

        # Gets the distance from each vector, find the closest k values and returns the mode of their labels
        nearest_labels, sorted_dist = self._get_closest(x, self.k)

        # get counts
        labels, cnts = np.unique(nearest_labels, return_counts=True)
        labels = labels[np.argsort(cnts[::-1])]

        if self.weight_type == 'inverse_distance':
            max_val = 0
            max_l = 0
            for l in labels:
                m = np.sum(1/(sorted_dist[nearest_labels == l])**2)
                if m > max_val:
                    max_val = m
                    max_l = l

            return max_l

        return labels[0]

    def _pred_regress(self, x):
        """
        Prediction Algorithm for regression data
        :param data:
        :param labels:
        :return:
        """
        # Gets the distance from each vector, find the closest k values and returns the mode of their labels
        nearest_labels, sorted_dist = self._get_closest(x, self.k)

        if self.weight_type == 'inverse_distance':
            weights = (1/sorted_dist**2)
            return np.sum(weights*nearest_labels) / np.sum(weights)
        else:
            return np.mean(nearest_labels)

    #Returns the Mean score given input data and labels
    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
                X (array-like): A 2D numpy array with data, excluding targets
                y (array-like): A 2D numpy array with targets
        Returns:
                score : float
                        Mean accuracy of self.predict(X) wrt. y.
        """
        y_ = self.predict(X)
        y.flatten()
        n = len(y)
        if self.label_type == "class":
            return sum(y_ == y) / n
        else:
            return np.sum((y_ - y)**2) / n


# ------------------ KD TREE Implementation ------------------

class KDTNode:
    def __init__(self, x):
        if(type(x)is not np.ndarray):
            raise TypeError("X must be an np.array")
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

    def has_no_children(self):
        """
        Checks to see if node has children or not
        :return: bool True if has NO children
        """
        if(self.left is None and self.right is None):
            return True
        else:
            return False

    def has_left(self):
        """
        Checks for left child
        :return: bool True if has left child
        """
        return True if self.left is not None else False

    def has_right(self):
        """
        Check for right child
        :return: bool: True if has right child
        """
        return True if self.right is not None else False

    def __str__(self):
        """
        Overrride to string function
        :return: String of Node
        """
        return "Array: " + str([x for x in self.value]) + " Pivot:" + str(self.pivot)


class KDTree:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
        """

        if(self.root is None):
            # If the tree is empty assign root and k value
            new_node = KDTNode(data)
            new_node.pivot = 0
            self.k = len(data)
            self.root = new_node
        else:
            if (len(data) != self.k):
                # The data dimensions needs to match the tree dimensions
                raise ValueError("Data must be the same dimension as", self.k)

            # Find the parent of the node to insert
            parent_node = self.find_insert_parent(self. root, data)

            # Create new node and give it the right pivot value
            new_node = KDTNode(data)
            new_node.pivot = (parent_node.pivot + 1) % self.k

            # Place the new node to the left or right of the parent node depending
            if(parent_node.value[parent_node.pivot] > new_node.value[parent_node.pivot]):
                parent_node.left = new_node
            else:
                parent_node.right = new_node

        return

    def find_insert_parent(self, cur_node, data):
        """
        Resursive Function to find the correct parent node to attach a new node to
        :param cur_node: KDTNode starts with root
        :param data: data of new node ot insert
        :return: KDTNode: the appropriate parent node to attach the new node to
        """
        if(set(cur_node.value) == set(data)):
            raise ValueError("Value already in the Tree")
        if(cur_node.has_no_children()):
            #Base Case
            return cur_node
        if(cur_node.has_right() and cur_node.value[cur_node.pivot] < data[cur_node.pivot]):
            # Go right
            return self.find_insert_parent(cur_node.right, data)
        if (cur_node.has_left() and cur_node.value[cur_node.pivot] > data[cur_node.pivot]):
            #Go left
            return self.find_insert_parent(cur_node.left, data)
        else:
            return cur_node


    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        cur_node = self.root
        def closest_euclid(cur_node, nearest, d):
            """
            Recursive function to find the closest node by the euclidean distance
            :param cur_node: KDTNode: Root
            :param nearest: KDTNode: Current closest node
            :param d: float: euclid distance of the current node
            :return: (KDTNode, float): The node closest to z and its distance from z
            """
            if cur_node is None:
                return nearest, d
            x = cur_node.value
            i = cur_node.pivot
            norm = la.norm(x - z)
            if(norm < d):
                nearest = cur_node
                d = norm
            if(z[i] < x[i]):
                # Go left based on pivot
                nearest, d = closest_euclid(cur_node.left, nearest, d)
                if (z[i] + d > x[i]):
                    nearest, d = closest_euclid(cur_node.right, nearest, d)
            else:
                # Go right based on pivot
                nearest, d = closest_euclid(cur_node.right, nearest, d)
                if (z[i] - d <= x[i]):
                    nearest, d = closest_euclid(cur_node.left, nearest, d)

            return nearest, d

        nearest, d = closest_euclid(cur_node, cur_node, la.norm(cur_node.value - z))
        return nearest.value, d

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)

class KNeighborsClassifier:
    def __init__(self, n_neighbors):
        if(type(n_neighbors) is not int):
            raise TypeError("Needs to be an integer")
        self.n_neighbors = n_neighbors
        self.tree = None
        self.labels = None

    def fit(self, X, y):
        """
        Takes in data and labels and saves them as attributes
        :param X: ((m,k) ndarray): a training set of m k-dimensional points.
        :param y: ((k, ) ndarray): labels.
        :return:
        """
        #Initialize a tree
        self.tree = KDTree(X)
        self.labels = y

        return

    def predict(self, z):
        """
        Returns the most common label of the closest neighbors to z
        :param z: (k) ndarray: target point to find the closest points to
        :return: float, most common label among the k closest nodes
        """
        distances, indices = self.tree.query(z, self.n_neighbors)
        # Gets the labels for each of the closest nodes and returns the most frequent
        if(self.n_neighbors == 1):
            return self.labels[indices]
        else:
            mode = stats.mode([self.labels[i] for i in indices])
            return mode.mode[0]
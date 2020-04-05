import numpy as np
from numpy import linalg as la
from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import cdist

class Cluster:

    def __init__(self, nodes, idx, link_type):
        """
        :param nodes: ndarray (number_points x features)
        :param link_type: 'single' or 'complete' (min or max)
        :param idx:
        """

        self.nodes = nodes
        self.node_idxs = [idx]
        self.centroid = np.mean(self.nodes, axis=0)
        self.link_type = link_type
        self.n = len(self.nodes)

    def combine(self, other):
        """
        Combines two clusters together
        :param other: Cluster
        :return:
        """
        self.nodes = np.vstack((self.nodes, other.nodes))
        self.n = len(self.nodes)
        self.add_node_idx(other.node_idxs)
        return self

    def add_node_idx(self, idxs):
        """
        Add node indexes to the cluster
        :param idxs: list of indexes
        :return:
        """
        self.node_idxs = self.node_idxs + idxs
        return

    def get_distance(self, other):
        """
        Gets the distance of one cluster to another
        :param other: Cluster, other cluster to get distance to
        :param type: 'single' or 'complete', (min or max distance)
        :return:
        """
        if self.link_type == 'single':
            return np.min(cdist(self.nodes, other.nodes))
        else:
            return np.max(cdist(self.nodes, other.nodes))

    def calculate_sse(self):
        """
        Calculates the centroids and SSE for the cluster
        :return:
        """
        self.centroid = np.mean(self.nodes, axis=0)
        self.sse = np.sum(la.norm(self.nodes - self.centroid, axis=1) ** 2)
        return self.sse


class HACClustering(BaseEstimator, ClusterMixin):

    def __init__(self, k=3, link_type='single'):  ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
        self.clusters = None
        self.centroids = None
        self.SSE = None

    def fit(self, X, y=None):
        """ Fit the data
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.clusters = np.array([Cluster(X[i:i + 1, :], i, self.link_type) for i in range(len(X))])

        while len(self.clusters) > self.k:
            # calculate distances
            distances = self.calculate_distances()
            minimum = np.min(distances[distances != 0])
            argmin = np.argwhere(distances == minimum)
            first = np.min(argmin)
            second = np.max(argmin)

            # combine the two closest clusters
            self.clusters[first] = self.clusters[first].combine(self.clusters[second])

            # remove second cluster
            self.clusters = np.delete(self.clusters, second)

        # calculate sse for each cluster
        self.SSE = np.array([c.calculate_sse() for c in self.clusters])

        return self

    def calculate_distances(self):
        """
        Calculate distances between all clusters
        :return:
        """
        clusters = self.clusters
        n = len(clusters)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = clusters[i].get_distance(clusters[j])
        return distances

    def save_clusters(self, filename):
        """
            Used for grading.
        """
        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        total_SSE = sum(self.SSE)
        f.write("{:.4f}\n\n".format(total_SSE))
        for i, cluster in enumerate(self.clusters):
            f.write(np.array2string(cluster.centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(cluster.n))
            f.write("{:.4f}\n\n".format(self.SSE[i]))
        f.close()
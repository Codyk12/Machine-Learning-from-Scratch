import numpy as np
from numpy import linalg as la
from sklearn.base import BaseEstimator, ClusterMixin


class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self, k=3, debug=False): ## add parameters here
        """
        :param k: how many final clusters to have
        :param debug: if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        :param starting_centroids:
        """
        self.k = k
        self.debug = debug
        self.clusters = {}
        self.centroids = None
        self.SSE = np.empty(k)

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        if self.debug:
            cur_centers = X[:self.k, :].copy()
        else:
            cur_centers = X[np.random.choice(range(len(X)), self.k, replace=False), :].copy()

        prev_centers = np.zeros_like(cur_centers)
        distances = np.empty((len(X), len(prev_centers)))

        cluster_idxs = np.arange(self.k)

        while (prev_centers != cur_centers).any():

            # calculate distances
            for i, center in enumerate(cur_centers):
                distances[:, i] = la.norm(X - center, axis=1)

            # calculate closest cluster label
            min_idx = distances.argmin(axis=1)

            # update centers
            prev_centers = cur_centers.copy()
            for idx in cluster_idxs:
                cur_centers[idx, :] = X[min_idx == idx, :].mean(axis=0)

        # store clusters and SSE
        for idx in cluster_idxs:
            cluster = X[min_idx == idx, :]
            self.clusters[idx] = cluster
            self.SSE[idx] = np.sum(la.norm(cluster - cur_centers[idx, :], axis=1)**2)

        self.centroids = cur_centers
        return self

    def save_clusters(self, filename):
        """
            Used for grading.
        """
        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        total_SSE = sum(self.SSE)
        f.write("{:.4f}\n\n".format(total_SSE))
        for k in self.clusters.keys():
            f.write(np.array2string(self.centroids[k,:], precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[k])))
            f.write("{:.4f}\n\n".format(self.SSE[k]))
        f.close()

import numpy as np
import sys

class Knn:
    def __init__(self, k=5):
        self.k = 5
        self.train_data = None
        self.train_target = None

    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target

    def predict(self, test_data):
        y = list()
        for params in test_data:
            distance = list()
            for i in range(len(self.train_data)):
                data = self.train_data[i]
                dist = 0
                for j in range(len(data)):
                    row = data[j]
                    dist += self.dtw_distance(params[j], row)
                distance.append(dist/8.0)
            indexs = np.argsort(np.array(distance), axis=0)[:self.k]
            labels = np.array([self.train_target[x] for x in indexs])
        y.append(np.argmax(np.bincount(labels)))
        return y

    def dtw_distance(self, ts_a, ts_b):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared

        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function

        Returns
        -------
        DTW distance between A and B
        """
        d = lambda x, y: abs(x - y)
        max_warping_window = 10000

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - max_warping_window),
                           min(N, i + max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window
        return cost[-1, -1]

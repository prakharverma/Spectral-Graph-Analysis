import random

from Spectral_Graph_Analysis.scripts.Graph import Graph


class CustomKmeans:
    def __init__(self, n_clusters, graph: Graph):
        self.k = n_clusters
        self.data = None
        self.clustered_data = {}
        self.mean_val = [0] * self.k
        self.graph = graph

    def get_optimal_cluster(self, d_pnt):
        pnt_neighbors = self.graph.get_neighbors(d_pnt)
        cluster_neighbor_count = [0] * self.k
        for c_key in self.clustered_data.keys():
            vals = self.clustered_data[c_key]
            for n in pnt_neighbors:
                if n in vals:
                    cluster_neighbor_count[c_key] += 1

        return cluster_neighbor_count.index(max(cluster_neighbor_count))

    def _distribute_data_into_clusters(self):
        self.clustered_data = {}

        for d in self.data:
            # skip mean data points
            if d in self.mean_val:
                continue

            self.clustered_data[self.get_optimal_cluster(d)].append(d)

        return True

    def fit(self, x, max_iter=300):
        self.data = x

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Random mean point initialization and adding it to the respective cluster
        for i in range(self.k):
            self.mean_val[i] = x[random.randint(0, len(x))]

        for i, mean_val in enumerate(self.mean_val):
            self.clustered_data[i] = [mean_val]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for _ in range(max_iter):
            self._distribute_data_into_clusters()

        # add first mean val data point to the cluster
        for i, mean_val in enumerate(self.mean_val):
            self.clustered_data[i] = [mean_val]

        return self.clustered_data

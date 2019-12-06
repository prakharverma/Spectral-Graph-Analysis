import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import lil_matrix


class Graph:
    def __init__(self):
        self.g = nx.Graph()

    def add_edges(self, edges_data):
        self.g.add_edges_from(edges_data)

    def get_edges_count(self):
        return self.g.number_of_edges()

    def get_nodes_count(self):
        return self.g.number_of_nodes()

    def get_adjacency_matrix(self):
        adj_matrix = lil_matrix((self.get_nodes_count(), self.get_nodes_count()))
        for edge in self.get_edges():
            adj_matrix[edge[0], edge[1]] = 1
            adj_matrix[edge[1], edge[0]] = 1

        return adj_matrix
        # return nx.adjacency_matrix(self.g, nodelist=range(0, self.g.number_of_nodes()))

    def get_diagonal_matrix(self):
        nodelist = list(self.g)
        A = nx.to_scipy_sparse_matrix(self.g, nodelist=nodelist, format='csr')
        n, m = A.shape
        diags = A.sum(axis=1).flatten()
        D = sparse.spdiags(diags.flatten(), [0], m, n, format='csr')
        return D

    def get_graph(self):
        return self.g

    def plot_graph(self):
        nx.draw(self.g)
        plt.draw()
        plt.show()

    def get_nodes(self):
        return self.g.nodes

    def get_neighbors(self, node):
        return self.g.neighbors(node)

    def get_max_degree_elements(self, n):
        degrees = self.g.degree
        degree_dict = {}
        for d in degrees:
            degree_dict[d[0]] = d[1]

        sorted_nodes_distance = [k for k in
                                 sorted(degree_dict, key=degree_dict.get, reverse=True)]

        return sorted_nodes_distance[:n]

    def get_edges(self):
        return self.g.edges

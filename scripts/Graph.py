import networkx as nx
import matplotlib.pyplot as plt


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
        return self.g.adjacency()

    def get_graph(self):
        return self.g

    def plot_graph(self):
        nx.draw(self.g)
        plt.draw()
        plt.show()

    def get_nodes(self):
        return self.g.nodes

    def get_laplacian(self, normalized=False):
        if normalized:
            return nx.normalized_laplacian_matrix(self.g)

        return nx.laplacian_matrix(self.g)

    def get_neighbors(self, node):
        return self.g.neighbors(node)

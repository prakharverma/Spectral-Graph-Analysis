from scipy.sparse import linalg
import numpy as np
from sklearn.cluster import KMeans

import Spectral_Graph_Analysis.scripts.utility as utils

if __name__ == '__main__':

    txt_file = "../../graphs_processed/ca-GrQc.txt"

    graph, k, header = utils.create_graph_from_txt(txt_file)

    print("Computing Laplacian...\n")
    laplacian = graph.get_laplacian(normalized=True)
    laplacian = laplacian.asfptype()

    print("Computing Eigen values...\n")
    eigen_values, eigen_vectors = linalg.eigs(laplacian, k=k)
    eigen_vectors = np.real(eigen_vectors)

    print("Perform k-means...\n")
    k = KMeans(n_clusters=k)
    k.fit(eigen_vectors)

    print("Finding the nodes cluster...\n")
    results = k.labels_
    cluster_nodes = {}
    nodes = list(graph.get_nodes())
    for i in range(graph.get_nodes_count()):
        if results[i] not in cluster_nodes.keys():
            cluster_nodes[results[i]] = [nodes[i]]
        else:
            cluster_nodes[results[i]].append(nodes[i])

    print("Creating a output file...\n")
    utils.create_output_file(cluster_nodes, "test.txt", header)

    print("Calculating objective function values...")
    objective_val = utils.get_objective_value(graph, cluster_nodes)
    print(objective_val)

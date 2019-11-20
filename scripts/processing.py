from scipy.sparse import linalg
import numpy as np
from sklearn.cluster import KMeans
import math
from sklearn.decomposition import TruncatedSVD, pca

import Spectral_Graph_Analysis.scripts.utility as utils

if __name__ == '__main__':

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameteres
    txt_file = "../../graphs_processed/ca-GrQc.txt"
    output_file = "ca-GrQc.txt"

    normalized_laplacian = True

    eigen_vectors = True
    normalize_eigen_vectors = True
    truncated_SVD = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    graph, k, header = utils.create_graph_from_txt(txt_file)

    print("Computing Laplacian...\n")
    laplacian = graph.get_laplacian(normalized=normalized_laplacian)
    laplacian = laplacian.asfptype()

    if eigen_vectors:
        print("Computing Eigen values...\n")
        eigen_values, eigen_vectors = linalg.eigs(laplacian, k=k, which="LR")
        eigen_vectors = np.real(eigen_vectors)

        # Normalizing the row. According to Luxburg06_TR.pdf
        if normalize_eigen_vectors:
            # FIXME : Optimize if it works
            for i in range(eigen_vectors.shape[0]):
                s = math.sqrt(sum(eigen_vectors[i]**2))
                for j in range(eigen_vectors.shape[1]):
                    eigen_vectors[i, j] = eigen_vectors[i, j] / s

    if truncated_SVD:
        svd = TruncatedSVD(n_components=k)
        eigen_vectors = svd.fit_transform(laplacian)

    print("Perform k-means...\n")
    k = KMeans(n_clusters=k)
    k.fit(eigen_vectors)

    print("Finding the nodes cluster...\n")
    results = k.labels_
    cluster_nodes = {}
    for node, cluster in enumerate(results):
        if cluster not in cluster_nodes.keys():
            cluster_nodes[cluster] = [node]
        else:
            cluster_nodes[cluster].append(node)

    for c_key in cluster_nodes.keys():
        print(f"Cluster {c_key} : {len(cluster_nodes[c_key])}")

    print("\nCreating a output file...\n")
    utils.create_output_file(cluster_nodes, output_file, header)

    print("Calculating objective function values...")
    objective_val = utils.get_objective_value(graph, cluster_nodes)
    print(objective_val)

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
import numpy as np


from Spectral_Graph_Analysis.scripts.Graph import Graph


def perform_clustering(graph: Graph, k: int, normalized_laplacian=True, eigen_vectors=True, normalize_eigen_vectors=True,
                       truncated_SVD=False):

    print("Computing Laplacian...\n")

    adjacency_matrix = graph.get_adjacency_matrix()
    diagonal_matrix = graph.get_custom_diagonal_matrix(adjacency_matrix)

    laplacian = diagonal_matrix - adjacency_matrix

    if normalized_laplacian:
        eigen_vectors = normalize(eigen_vectors)

    if eigen_vectors:
        print("Computing Eigen values...\n")
        eigen_values, eigen_vectors = linalg.eigs(laplacian, M=diagonal_matrix, k=k, which="SR")
        eigen_vectors = eigen_vectors.real

        if normalize_eigen_vectors:
            eigen_vectors = normalize(eigen_vectors)

    if truncated_SVD:
        svd = TruncatedSVD(n_components=k)
        eigen_vectors = svd.fit_transform(laplacian)
        if normalize_eigen_vectors:
            eigen_vectors = normalize(eigen_vectors)

    print("Perform k-means...\n")

    # TESTING kmeans initialization
    # kmeans_initial = np.array(graph.get_max_degree_elements(k)).reshape((5, 1))

    kmeans = KMeans(n_clusters=k)
    results = kmeans.fit_predict(eigen_vectors)
    cluster_centroids = kmeans.cluster_centers_
    transformed_x = kmeans.transform(eigen_vectors)

    # c_kmeans = CustomKmeans(k, graph)

    print("Finding the nodes cluster...\n")
    cluster_nodes = {}
    for node, cluster in enumerate(results):
        if cluster not in cluster_nodes.keys():
            cluster_nodes[cluster] = [node]
        else:
            cluster_nodes[cluster].append(node)

    return cluster_nodes, cluster_centroids, transformed_x

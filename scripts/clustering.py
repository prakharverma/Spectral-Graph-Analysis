from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse

from Graph import Graph


def get_custom_diagonal_matrix(adjacency, laplacian_algo):
    diag_matrix = adjacency.sum(axis=1)
    diag_matrix = np.array(diag_matrix.reshape((len(diag_matrix), )))
    diag_matrix = diag_matrix[0]

    if laplacian_algo == "SNL":
        diag_matrix = sparse.diags(np.power(diag_matrix, -.5))
    elif laplacian_algo == "UL" or laplacian_algo == "SM":
        diag_matrix = sparse.diags(diag_matrix)
    elif laplacian_algo == "RWL":
        diag_matrix = sparse.diags(1/diag_matrix)

    return diag_matrix


def calculate_laplacian(graph: Graph, laplacian_algo):
    adjacency_matrix = graph.get_adjacency_matrix()
    diagonal_matrix = get_custom_diagonal_matrix(adjacency_matrix, laplacian_algo)

    if laplacian_algo == "SNL":
        laplacian = sparse.identity(adjacency_matrix.shape[0]) - (diagonal_matrix * adjacency_matrix * diagonal_matrix)

    if laplacian_algo == "RWL":
        laplacian = sparse.identity(adjacency_matrix.shape[0]) - (diagonal_matrix * adjacency_matrix)

    elif laplacian_algo == "UL" or laplacian_algo == "SM":
        laplacian = diagonal_matrix - adjacency_matrix

    return laplacian, adjacency_matrix, diagonal_matrix


def kmeans(vals, k, intialize_kmeans=False, graph=None, eigen_vectors=None):
    if intialize_kmeans:
        kmeans_centroid = graph.get_max_degree_elements(k)
        kmeans_initial = np.zeros(shape=(k, k))

        for i, cen in enumerate(kmeans_centroid):
           kmeans_initial[i, :] = eigen_vectors[cen]
        kmeans = KMeans(n_clusters=k, init=kmeans_initial, random_state=1)

    else:
        kmeans = KMeans(n_clusters=k)

    results = kmeans.fit_predict(vals)
    cluster_centroids = kmeans.cluster_centers_
    transformed_x = kmeans.transform(vals)

    return results, cluster_centroids, transformed_x


def perform_spectral_clustering(graph: Graph,
                                k: int,
                                calculate_eigen_vectors=True,
                                normalize_eigen_vectors=True,
                                truncated_SVD=False,
                                laplacian_algo="UL"):

    assert laplacian_algo in ["UL", "SNL", "RWL", "SM"]

    print("Computing Laplacian...\n")

    laplacian, adjacency_matrix, diagonal_matrix = calculate_laplacian(graph, laplacian_algo)

    if calculate_eigen_vectors:
        print("Computing Eigen values...\n")
        if laplacian_algo == "SM":
            eigen_values, eigen_vectors = linalg.eigs(laplacian, M=diagonal_matrix, k=k, which="SR")

        else:
            eigen_values, eigen_vectors = linalg.eigs(laplacian, k=k, which="SR")

        eigen_vectors = eigen_vectors.real

    elif truncated_SVD:
        svd = TruncatedSVD(n_components=k)
        eigen_vectors = svd.fit_transform(laplacian)

    if normalize_eigen_vectors:
        eigen_vectors = normalize(eigen_vectors, norm="l2", axis=1)

    print("Perform k-means...\n")

    results, cluster_centroids, transformed_x = kmeans(eigen_vectors, k)

    print("Finding the nodes cluster...\n")
    cluster_nodes = {}
    for node, cluster in enumerate(results):
        if cluster not in cluster_nodes.keys():
            cluster_nodes[cluster] = [node]
        else:
            cluster_nodes[cluster].append(node)

    return cluster_nodes, cluster_centroids, transformed_x

from sklearn.cluster import KMeans
from scipy.sparse import linalg
from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse
import copy

from Graph import Graph
import utility as utils


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
                                neighbor_list,
                                normalize_eigen_vectors=True,
                                laplacian_algo="UL"):

    assert laplacian_algo in ["UL", "SNL", "RWL", "SM"]

    # FIXME : Change 4
    eigens_to_calculate = 20
    initial_eigens_fixed = 2

    print("Computing Laplacian...\n")

    laplacian, adjacency_matrix, diagonal_matrix = calculate_laplacian(graph, laplacian_algo)

    print("Computing Eigen values...\n")
    if laplacian_algo == "SM":
        # FIXME : Change 3
        eigen_values, eigen_vectors = linalg.eigs(laplacian, M=diagonal_matrix, k=eigens_to_calculate, which="SR")

    else:
        # FIXME : Change 1
        eigen_values, eigen_vectors = linalg.eigs(laplacian, k=eigens_to_calculate, which="SM")

    eigen_vectors = eigen_vectors.real

    if normalize_eigen_vectors:
        eigen_vectors = normalize(eigen_vectors, norm="l2", axis=1)

    print("Perform k-means...\n")

    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FIXME : Change 1
    best_obj_val = 99999999
    cluster_nodes, cluster_centroids, transformed_x = None, None, None
    for _ in range(100):
        eigen_vects = copy.deepcopy(eigen_vectors)
        random_eigen_index = np.random.randint(initial_eigens_fixed, eigens_to_calculate, k)
        eigen_vec = np.zeros((graph.get_nodes_count(), k))

        for h in range(initial_eigens_fixed):
            eigen_vec[:, h] = eigen_vects[:, h]

        for h in range(initial_eigens_fixed, k):
            eigen_vec[:, h] = eigen_vects[:, random_eigen_index[h]]

        t_results, t_cluster_centroids, t_transformed_x = kmeans(eigen_vec, k)

        cluster_nodes_t = {}
        for node, cluster in enumerate(t_results):
            if cluster not in cluster_nodes_t.keys():
                cluster_nodes_t[cluster] = [node]
            else:
                cluster_nodes_t[cluster].append(node)

        obj_val = utils.get_objective_value(cluster_nodes_t, copy.deepcopy(neighbor_list))

        if obj_val < best_obj_val:
            best_obj_val = obj_val
            cluster_nodes = cluster_nodes_t
            cluster_centroids = t_cluster_centroids
            transformed_x = t_transformed_x

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # results, cluster_centroids, transformed_x = kmeans(eigen_vectors, k)

    # print("Finding the nodes cluster...\n")
    # cluster_nodes = {}
    # for node, cluster in enumerate(results):
    #     if cluster not in cluster_nodes.keys():
    #         cluster_nodes[cluster] = [node]
    #     else:
    #         cluster_nodes[cluster].append(node)

    return cluster_nodes, cluster_centroids, transformed_x

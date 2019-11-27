import utility as utils
import clustering
import cluster_optimize as optimize

if __name__ == '__main__':

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameteres
    txt_file = "../../graphs_processed/ca-GrQc.txt"
    output_file = "../../output/ca-GrQc.output"

    laplacian_algo = "UL"  # {Symmetric Normalized Laplacian=SNL; Unnormalized Laplacian=UL; Random Walk Laplacian=RWL}
    eigen_vectors = True
    normalize_eigen_vectors = True

    truncated_SVD = False

    perform_greedy_algo = False
    calculate_objective_value = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    graph, k, header = utils.create_graph_from_txt(txt_file)

    cluster_nodes, cluster_centroids, transformed_x = clustering.perform_spectral_clustering(graph,
                                                                                             k,
                                                                                             calculate_eigen_vectors=eigen_vectors,
                                                                                             normalize_eigen_vectors=normalize_eigen_vectors,
                                                                                             truncated_SVD=truncated_SVD,
                                                                                             laplacian_algo=laplacian_algo)

    if calculate_objective_value:
        print("Calculating objective function values...")
        objective_val = utils.get_objective_value(graph, cluster_nodes)
        print(objective_val)

    if perform_greedy_algo:
        print("Executing Greedy Algorithm...")
        cluster_nodes = optimize.greedy_algorithms(objective_val, cluster_nodes, graph, iters=2)

    print("Creating a output file...\n")
    utils.create_output_file(cluster_nodes, output_file, header)

    if calculate_objective_value:
        print("Calculating final objective value")
        objective_val = utils.get_objective_value(graph, cluster_nodes)
        print(objective_val)


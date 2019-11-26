import Spectral_Graph_Analysis.scripts.utility as utils
import Spectral_Graph_Analysis.scripts.cluster_optimize as optimize
import Spectral_Graph_Analysis.scripts.clustering as clustering
import copy

if __name__ == '__main__':

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameteres
    txt_file = "../../graphs_processed/Oregon-1.txt"
    output_file = "../../output/Oregon-1.output"

    normalized_laplacian = False

    eigen_vectors = True
    normalize_eigen_vectors = False
    truncated_SVD = False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    graph, k, header = utils.create_graph_from_txt(txt_file)

    cluster_nodes, cluster_centroids, transformed_x = clustering.perform_clustering(graph,
                                                                                    k,
                                                                                    normalized_laplacian=normalized_laplacian,
                                                                                    eigen_vectors=eigen_vectors,
                                                                                    normalize_eigen_vectors=normalize_eigen_vectors,
                                                                                    truncated_SVD=truncated_SVD)

    print("Calculating objective function values...")
    objective_val = utils.get_objective_value(graph, cluster_nodes)
    print(objective_val)

    # GREDDY ALGORITHM BELOW
    # for _ in range(2):
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Optimizing by finding nodes with maximum edges in the other cluster
    #     previous_objective_val = objective_val
    #     current_objective = -1
    #     optimize_cluster_node = copy.deepcopy(cluster_nodes)
    #     while current_objective < previous_objective_val:
    #
    #         cluster_nodes = copy.deepcopy(optimize_cluster_node)
    #         if current_objective != -1:
    #             previous_objective_val = current_objective
    #             objective_val = current_objective
    #         else:
    #             previous_objective_val = objective_val
    #
    #         biggest_cluster_nodes, biggest_cluster_id = optimize.get_biggest_cluster_nodes(optimize_cluster_node)
    #
    #         points_to_move = optimize.get_nodes_with_more_outward_edges(graph, biggest_cluster_nodes)
    #
    #         # putting the nodes into nodes to move and removing them from optimize cluster
    #         nodes_to_move_id = []
    #         for pnt in points_to_move:
    #             nodes_to_move_id.append(pnt[0])
    #             if pnt[0] in optimize_cluster_node[biggest_cluster_id]:
    #                 optimize_cluster_node[biggest_cluster_id].remove(pnt[0])
    #
    #         if biggest_cluster_id == 0:
    #             optimize_cluster_node[1] += nodes_to_move_id
    #         else:
    #             optimize_cluster_node[0] += nodes_to_move_id
    #
    #         for c_key in optimize_cluster_node.keys():
    #             print(f"Cluster {c_key} : {len(optimize_cluster_node[c_key])}")
    #
    #         print("\nCalculating objective function values...")
    #         current_objective = utils.get_objective_value(graph, optimize_cluster_node)
    #         print(current_objective)
    #
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    #     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #     # Optimizing by finding nodes with minimum edges inside cluster
    #     previous_objective_val = objective_val
    #     current_objective = -1
    #     optimize_cluster_node = copy.deepcopy(cluster_nodes)
    #     while current_objective < previous_objective_val:
    #         cluster_nodes = copy.deepcopy(optimize_cluster_node.copy())
    #         if current_objective != -1:
    #             previous_objective_val = current_objective
    #             objective_val = current_objective
    #         else:
    #             previous_objective_val = objective_val
    #
    #         biggest_cluster_nodes, biggest_cluster_id = optimize.get_biggest_cluster_nodes(optimize_cluster_node)
    #
    #         points_to_move = optimize.get_points_with_min_inside_edges(graph, biggest_cluster_nodes, 100)
    #
    #         nodes_to_move_id = []
    #         for pnt in points_to_move:
    #             nodes_to_move_id.append(pnt[0])
    #             if pnt[0] in optimize_cluster_node[biggest_cluster_id]:
    #                 optimize_cluster_node[biggest_cluster_id].remove(pnt[0])
    #
    #         if biggest_cluster_id == 0:
    #             optimize_cluster_node[1] += nodes_to_move_id
    #         else:
    #             optimize_cluster_node[0] += nodes_to_move_id
    #
    #         for c_key in optimize_cluster_node.keys():
    #             print(f"Cluster {c_key} : {len(optimize_cluster_node[c_key])}")
    #
    #         print("\nCalculating objective function values...")
    #         current_objective = utils.get_objective_value(graph, optimize_cluster_node)
    #         print(current_objective)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print("Creating a output file...\n")
    utils.create_output_file(cluster_nodes, output_file, header)

    print("Final objective value")
    print(utils.get_objective_value(graph, cluster_nodes))

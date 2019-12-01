import utility as utils
import clustering
import cluster_optimize as optimize
import os
import copy


def process_file(txt_file_path, output_dir, laplacian_algo, normalize_eigen_vectors, perform_greedy_algo, calculate_objective_value=False):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Output Directory already exist")
        return False

    graph, k, header = utils.create_graph_from_txt(txt_file_path)
    neighbor_list = utils.get_neighbor_list(graph)

    file_name = txt_file_path.split("/")[-1].split(".")[0]

    # do it for all the options
    for lap_algo in laplacian_algo:
        for normalize in normalize_eigen_vectors:
            for greedy in perform_greedy_algo:
                output_file = os.path.join(output_dir, file_name + "_" + str(lap_algo) + "_" + str(normalize) + "_" + str(greedy))

                cluster_nodes, cluster_centroids, transformed_x = clustering.perform_spectral_clustering(graph,
                                                                                                         k,
                                                                                                         normalize_eigen_vectors=normalize,
                                                                                                         laplacian_algo=lap_algo)

                objective_val = -1
                if calculate_objective_value:
                    print("Calculating objective function values...")
                    objective_val = utils.get_objective_value(cluster_nodes, copy.deepcopy(neighbor_list))
                    print(objective_val)

                if greedy:
                    print("Executing Greedy Algorithm...")
                    cluster_nodes = optimize.greedy_algorithms(objective_val, cluster_nodes, graph, copy.deepcopy(neighbor_list), iters=2)

                    if calculate_objective_value:
                        print("Calculating final objective value")
                        objective_val = utils.get_objective_value(cluster_nodes, copy.deepcopy(neighbor_list))
                        print(objective_val)

                output_file = output_file + "_" + str(round(objective_val, 4)) + ".txt"

                print("Creating a output file...\n")
                utils.create_output_file(cluster_nodes, output_file, header)

    return True


if __name__ == '__main__':

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameteres
    txt_file = "../../graphs_processed/ca-GrQc.txt"
    output_dir = "../../output/ca-GrQc"

    laplacian_algo = ["RWL", "SNL", "SM", "UL"]
    normalize_eigen_vectors = [False, True]
    perform_greedy_algo = [True, False]
    calculate_objective_value = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    process_file(txt_file, output_dir, laplacian_algo, normalize_eigen_vectors, perform_greedy_algo, calculate_objective_value)

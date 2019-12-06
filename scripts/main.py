import utility as utils
import clustering
import cluster_optimize as optimize
import os
import copy


def process_file(txt_file_path,
                 output_dir,
                 laplacian_algo,
                 normalize_eigen_vectors,
                 perform_greedy_algo,
                 calculate_objective_value=False,
                 n_kmeans=1,
                 eigens_to_calculate=-1,
                 fixed_eigens=-1):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Output Directory already exist")
        return False

    graph, k, header = utils.create_graph_from_txt(txt_file_path)
    neighbor_list = utils.get_neighbor_list(graph)

    if eigens_to_calculate == -1:
        eigens_to_calculate = k

    if fixed_eigens == -1:
        fixed_eigens = k

    file_name = txt_file_path.split("/")[-1].split(".")[0]

    # do it for all the options
    for lap_algo in laplacian_algo:
        for normalize in normalize_eigen_vectors:
            for greedy in perform_greedy_algo:
                output_file = os.path.join(output_dir, file_name + "_" + str(lap_algo) + "_" + str(normalize) + "_" + str(greedy))

                cluster_nodes, cluster_centroids, transformed_x = clustering.perform_spectral_clustering(graph,
                                                                                                         k,
                                                                                                         normalize_eigen_vectors=normalize,
                                                                                                         laplacian_algo=lap_algo,
                                                                                                         neighbor_list=neighbor_list,
                                                                                                         n_kmeans=n_kmeans,
                                                                                                         eigens_calculate=eigens_to_calculate,
                                                                                                         fixed_eigens=fixed_eigens
                                                                                                         )

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
    import time
    start_time = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameteres
    txt_file = "../../graphs_processed/soc-Epinions1.txt"
    output_dir = "../../output/soc-Epinions1_test"

    laplacian_algo = ["SM", "SNL", "RWL", "UL"]  # SM, RWL, UL, SNL
    normalize_eigen_vectors = [False, True]  # True, False
    perform_greedy_algo = [False, True]  # True, False
    calculate_objective_value = True  # True, False

    n_kmeans = 1
    eigens_to_calculate = -1  # -1 means to calculate k eigen values
    fixed_eigens = -1  # -1 means to set this to k

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    process_file(txt_file,
                 output_dir,
                 laplacian_algo,
                 normalize_eigen_vectors,
                 perform_greedy_algo,
                 calculate_objective_value,
                 n_kmeans,
                 eigens_to_calculate,
                 fixed_eigens
                 )

    print(f"Time Taken : {round(time.time() - start_time,3)} sec")


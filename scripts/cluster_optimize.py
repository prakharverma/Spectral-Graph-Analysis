import math
import copy

import utility as utils


def get_cluster(cluster_info, node):
    for key in cluster_info.keys():
        if node in cluster_info[key]:
            return key


def get_biggest_cluster_nodes(cluster_info):
    max_items = 0
    biggest_cluster = None
    for key in cluster_info.keys():
        if max_items < len(cluster_info[key]):
            max_items = len(cluster_info[key])
            biggest_cluster = key

    return cluster_info[biggest_cluster], biggest_cluster


def get_distance(x, y):
    return math.sqrt((x[0] - x[1])**2 + (y[0]-y[1])**2)


def get_farthest_points_from_centroid(centroid, points_list, transformed_data, n):
    nodes_distance = {}
    for i, node_id in enumerate(points_list):
        nodes_distance[node_id] = get_distance(centroid, transformed_data[node_id])

    sorted_nodes_distance = [(k, nodes_distance[k]) for k in sorted(nodes_distance, key=nodes_distance.get, reverse=True)]

    return sorted_nodes_distance[:n]


def get_points_with_min_inside_edges(graph, points_list, n):
    nodes_connections = {}
    for pnt in points_list:
        count = 0
        for neighbor in graph.get_neighbors(pnt):
            # FIXME : list should be updated as nodes are being moved into different clusters
            if neighbor in points_list:
                count += 1

        nodes_connections[pnt] = count

    sorted_nodes_distance = [(k, nodes_connections[k]) for k in sorted(nodes_connections, key=nodes_connections.get)]

    return sorted_nodes_distance[:n]


def get_nodes_with_more_outward_edges(graph, points_list):
    nodes_connections = {}
    same_cluster_nodes = points_list.copy()
    for pnt in points_list:
        inward_count = 0
        outward_count = 0
        for neighbor in graph.get_neighbors(pnt):
            if neighbor in same_cluster_nodes:
                inward_count += 1
            else:
                outward_count += 1

        if outward_count >= inward_count:
            nodes_connections[pnt] = 1
            same_cluster_nodes.remove(pnt)

    return [(k, nodes_connections[k]) for k in nodes_connections.keys()]


def greedy_algorithms(objective_val, cluster_nodes, graph, iters=2):
    for _ in range(iters):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Optimizing by finding nodes with maximum edges in the other cluster
        previous_objective_val = objective_val
        current_objective = -1
        optimize_cluster_node = copy.deepcopy(cluster_nodes)
        while current_objective < previous_objective_val:

            cluster_nodes = copy.deepcopy(optimize_cluster_node)
            if current_objective != -1:
                previous_objective_val = current_objective
                objective_val = current_objective
            else:
                previous_objective_val = objective_val

            biggest_cluster_nodes, biggest_cluster_id = get_biggest_cluster_nodes(optimize_cluster_node)

            points_to_move = get_nodes_with_more_outward_edges(graph, biggest_cluster_nodes)

            # putting the nodes into nodes to move and removing them from optimize cluster
            nodes_to_move_id = []
            for pnt in points_to_move:
                nodes_to_move_id.append(pnt[0])
                if pnt[0] in optimize_cluster_node[biggest_cluster_id]:
                    optimize_cluster_node[biggest_cluster_id].remove(pnt[0])

            if biggest_cluster_id == 0:
                optimize_cluster_node[1] += nodes_to_move_id
            else:
                optimize_cluster_node[0] += nodes_to_move_id

            for c_key in optimize_cluster_node.keys():
                print(f"Cluster {c_key} : {len(optimize_cluster_node[c_key])}")

            print("\nCalculating objective function values...")
            current_objective = utils.get_objective_value(graph, optimize_cluster_node)
            print(current_objective)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Optimizing by finding nodes with minimum edges inside cluster
        previous_objective_val = objective_val
        current_objective = -1
        optimize_cluster_node = copy.deepcopy(cluster_nodes)
        while current_objective < previous_objective_val:
            cluster_nodes = copy.deepcopy(optimize_cluster_node.copy())
            if current_objective != -1:
                previous_objective_val = current_objective
                objective_val = current_objective
            else:
                previous_objective_val = objective_val

            biggest_cluster_nodes, biggest_cluster_id = get_biggest_cluster_nodes(optimize_cluster_node)

            points_to_move = get_points_with_min_inside_edges(graph, biggest_cluster_nodes, 100)

            nodes_to_move_id = []
            for pnt in points_to_move:
                nodes_to_move_id.append(pnt[0])
                if pnt[0] in optimize_cluster_node[biggest_cluster_id]:
                    optimize_cluster_node[biggest_cluster_id].remove(pnt[0])

            if biggest_cluster_id == 0:
                optimize_cluster_node[1] += nodes_to_move_id
            else:
                optimize_cluster_node[0] += nodes_to_move_id

            for c_key in optimize_cluster_node.keys():
                print(f"Cluster {c_key} : {len(optimize_cluster_node[c_key])}")

            print("\nCalculating objective function values...")
            current_objective = utils.get_objective_value(graph, optimize_cluster_node)
            print(current_objective)

    return cluster_nodes

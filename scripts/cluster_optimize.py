from Spectral_Graph_Analysis.scripts.Graph import Graph
from scipy.sparse import linalg
import math
from sklearn.cluster import KMeans


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

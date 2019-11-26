from Spectral_Graph_Analysis.scripts import Graph


def create_graph_from_txt(txt_file_path: str):
    g = Graph.Graph()

    edges = []
    k = -1
    validation_total_edges = -1
    validation_total_nodes = -1
    header = ""

    print(f"Reading : {txt_file_path}")
    with open(txt_file_path) as file:
        line = file.readline()
        while line:
            if "#" not in line:
                line = line.replace("\n", "")
                val = line.split(" ")
                edges.append((int(val[0]), int(val[1])))
            else:
                header = line
                line = line.replace("\n", "")
                val = line.split(" ")
                validation_total_nodes = int(val[2])
                validation_total_edges = int(val[3])
                k = int(val[4])

            line = file.readline()

    print(f"File : {txt_file_path} : successfully processed")

    print(f"Total edges : {len(edges)}\n")

    print("Adding edges to graph\n")
    g.add_edges(edges)

    print("Graph data")
    print(f"Edges: {g.get_edges_count()}")
    print(f"Nodes: {g.get_nodes_count()}\n")

    print("Validating...")
    assert (g.get_edges_count() == validation_total_edges) and (g.get_nodes_count() == validation_total_nodes)
    print("Validation successful\n")

    print(f"Number of clusters : {k}")

    return g, k, header


def create_output_file(data: dict, output_path: str, header: str = None):
    with open(output_path, "w") as file:
        if header:
            file.write(header)
        for key in data.keys():
            for val in data[key]:
                file.write(f"{val}  {key}\n")
    return True


def get_objective_value(graph: Graph, cluster_nodes: dict):
    clusters = cluster_nodes.keys()
    objective_val = 0

    # FIXME : Optimize this
    for c in clusters:
        numerator = 0
        all_nodes_cluster = cluster_nodes[c]

        for val in all_nodes_cluster:
            neighbors = graph.get_neighbors(val)
            for n in neighbors:
                if n not in all_nodes_cluster:
                    numerator += 1

        objective_val += (numerator/len(all_nodes_cluster))

    return objective_val


def create_graph_from_edges(edges_list):
    g = Graph.Graph()
    g.add_edges(edges_list)
    return g

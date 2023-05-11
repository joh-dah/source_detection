import networkx as nx
from matplotlib import pyplot as plt
import torch


def one_hot_encode(label_list, n_diff_features):
    label_tensor = torch.tensor(label_list)
    return torch.nn.functional.one_hot(label_tensor, n_diff_features).float()


def plot_graph_with_status(g, status):
    colors = get_colors_for_infection_status(status)
    seed = 31
    pos = nx.spring_layout(g, seed=seed)
    plt.figure(figsize=(6, 4))
    nx.draw(g, pos=pos, with_labels=True, node_color=colors, node_size=250)


def get_colors_for_infection_status(infection_status):
    colors = []
    for _, i in infection_status.items():
        if i == 0:
            colors.append("blue")
        elif i == 1:
            colors.append("red")
        elif i == 2:
            colors.append("gray")
    return colors


def print_graph_info(graph):
    print("Directed graph:", graph.is_directed())
    print("Number of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

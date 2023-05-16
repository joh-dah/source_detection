import networkx as nx
from matplotlib import pyplot as plt
from GNN_model import get_ranked_source_predictions
from torch_geometric.utils.convert import from_networkx
import data_creation as dc
import torch
import constants as const
from pathlib import Path


def hex_to_RGB(hex):
    """ "#FFFFFF" -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    """[255,255,255] -> "#FFFFFF" """
    # Components need to be integers for hex to make sense
    RGB = [int(x) for x in RGB]
    return "#" + "".join(
        ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB]
    )


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF")"""
    # Starting and ending colors in RGB form
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    # Initilize a list of the output colors with the starting color
    RGB_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)
        ]
        # Add it to our list of output colors
        RGB_list.append(curr_vector)

    return [RGB_to_hex(RGB) for RGB in RGB_list]


def plot_predictions(net, n):
    for i in range(n):
        g = dc.create_graph()
        m = dc.model_SIR_signal_propagation(g)
        graph_structure = from_networkx(g)
        features = one_hot_encode(list(m.status.values()), const.N_FEATURES)

        plot_graph_with_status(g, m.initial_status, f"initial_{i}")
        plot_graph_with_status(g, m.status, f"current_{i}")

        Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)
        seed = 31
        pos = nx.spring_layout(g, seed=seed)
        plt.figure(figsize=(6, 4))

        ranked_predictions = get_ranked_source_predictions(
            net, features, graph_structure.edge_index
        )

        color_gradient = linear_gradient("#FF0000", "#0000FF", const.N_NODES)
        colors = [
            color_gradient[ranked_predictions.tolist().index(j)]
            for j in range(const.N_NODES)
        ]

        nx.draw(g, pos=pos, with_labels=True, node_color=colors, node_size=250)
        plt.savefig(f"{const.FIGURES_PATH}/prediction_{i}.png")


def one_hot_encode(label_list, n_diff_features):
    label_tensor = torch.tensor(label_list)
    return torch.nn.functional.one_hot(label_tensor, n_diff_features).float()


def plot_graph_with_status(g, status, title):
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    colors = get_colors_for_infection_status(status)
    seed = 31
    # pos = nx.spring_layout(g, seed=seed)
    pos = nx.circular_layout(g)
    plt.figure(figsize=(6, 4))
    nx.draw(g, pos=pos, with_labels=True, node_color=colors, node_size=50)
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")


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

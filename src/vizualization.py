import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.utils.convert import from_networkx
import src.data_creation as dc
from pathlib import Path
import src.constants as const
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_graph_with_status(g, colors, title, layout="spring"):
    """
    Plots graph and colors nodes based on their infection status.
    :param g: networkx graph
    :param status: dict with node as key and infection status as value
    :param title: title of the plot
    :param layout: layout of the plot
    """
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    seed = const.SEED
    if layout == "spring":
        pos = nx.spring_layout(g, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(g, seed=seed)
    else:
        raise AssertionError("Unknown layout")

    plt.figure(figsize=(6, 4))
    nx.draw(g, pos=pos, with_labels=True, node_color=colors, node_size=150)
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()

def plot_matching_graph(graph, matching, new_edges, title="matching_graph", layout="spring"):
    '''
    Plot the matching graph.
    :param graph: The matching graph.
    :param matching: Minimum weight matching between sources and predicted sources.
    :param new_edges: The minimum weight adjacent edge for each unmatched node.
    :param title: The title of the plot.
    :param layout: The layout of the plot.
    '''
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    seed = const.SEED
    if layout == "spring":
        pos = nx.spring_layout(graph, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(graph, seed=seed)
    else:
        raise AssertionError("Unknown layout")

    plt.figure(figsize=(6, 4))
    edge_colors = ["green" if edge in matching else "red" if edge in new_edges else "black" for edge in graph.edges]
    colors = ["red" if node[0] == "s" else "blue" for node in graph.nodes]
    nx.draw(graph, pos=pos, with_labels=True, node_color=colors, edge_color=edge_colors, node_size=150)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, "weight"))
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()


def plot_roc_curve(false_positives, true_positives):
    '''
    Plot ROC curves.
    :param false_positives: The false positives rates.
    :param true_positives: The true positives rates.
    '''
    print("Visualize ROC curve:")
    for i, false_positive in tqdm(enumerate(false_positives)):
        plt.plot(false_positive, true_positives[i])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(f"{const.FIGURES_PATH}/roc_curve__{i}.png")
        plt.close()

def get_colors_for_infection_status(infection_status):
    """
    Returns colors for nodes based on their infection status.
    :param infection_status: dict with node as key and infection status as value
    :return: list of colors
    """
    colors = []
    for _, i in infection_status.items():
        if i == 0:
            colors.append("blue")
        elif i == 1:
            colors.append("red")
        elif i == 2:
            colors.append("gray")
    return colors


def hex_to_RGB(hex):
    """
    Convert hex color to RGB.
    e.g: "#FFFFFF" -> [255,255,255]
    """
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def RGB_to_hex(RGB):
    """
    Convert RGB to hex color.
    [255,255,255] -> "#FFFFFF"
    """
    RGB = [int(x) for x in RGB]
    return "#" + "".join(
        ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in RGB]
    )


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
    """
    returns a gradient list of (n) colors between two hex colors.
    start_hex and finish_hex should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF")
    """
    s = hex_to_RGB(start_hex)
    f = hex_to_RGB(finish_hex)
    RGB_list = [s]
    for t in range(1, n):
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)
        ]
        RGB_list.append(curr_vector)
    return [RGB_to_hex(RGB) for RGB in RGB_list]


def plot_predictions(
    prop_model, ranked_predictions, title, layout="spring", colored_nodes=7
):
    """
    Plots the initial and the current graph with their infection status.
    Additionally, a graph is plotted with the colors of the nodes representing the predicted likelihood of beeing source.
    :param prop_models: list of propagation models
    :param ranked_predictions: list of ranked predictions
    :param layout: layout of the plot
    """
    g = nx.Graph()
    g.add_nodes_from(prop_model.graph.nodes)
    g.add_edges_from(prop_model.graph.edges)

    colors = get_colors_for_infection_status(prop_model.initial_status)
    plot_graph_with_status(g, colors, f"initial_{title}", layout)

    colors = get_colors_for_infection_status(prop_model.status)
    plot_graph_with_status(g, colors, f"current_{title}", layout)

    colors = ["#0000FF"] * const.N_NODES
    color_gradient = linear_gradient("#FF0000", "#0000FF", colored_nodes)
    for i, node in enumerate(ranked_predictions[:colored_nodes]):
        colors[node] = color_gradient[i]

    plot_graph_with_status(g, colors, f"prediction_{title}.png", layout)

from typing import Union
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from torch_geometric.utils.convert import from_networkx
import src.data_creation as dc
from pathlib import Path
import src.constants as const
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from ndlib.models.DiffusionModel import DiffusionModel


def plot_graph_with_colors(
    g: nx.Graph,
    node_values: np.ndarray,
    max_colored_value: int,
    title: str,
    cmap: Union[Colormap, str] = "viridis",
    layout: any = nx.kamada_kawai_layout,
):
    """
    Plots graph and colors nodes according to node_values.
    :param g: graph
    :param node_values: values for nodes
    :param max_colored_value: highest node value that should be mapped to a unique color
    :param title: title of plot
    :param cmap: colormap to use
    :param layout: graph plotting layout to use
    """
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    pos = layout(g)

    plt.figure(figsize=(6, 4))

    node_values = np.clip(node_values, 0, max_colored_value)
    nx.draw(
        g,
        pos=pos,
        with_labels=True,
        node_color=node_values,
        node_size=150,
        cmap=cmap,
        vmin=0,
        vmax=max_colored_value,
    )
    # add colorbar with custom int ticks
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_colored_value)
    )
    colorbar = plt.colorbar(sm, shrink=0.9)
    colorbar.set_ticks(np.arange(0, max_colored_value + 1, 1))
    colorbar.set_ticklabels(np.arange(0, max_colored_value + 1, 1))

    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()


def plot_matching_graph(g, matching, new_edges, title="matching_graph"):
    """Plots the matching graph to debug the min-matching distance metric"""
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    pos = nx.kamada_kawai_layout(g)

    plt.figure(figsize=(6, 4))
    edge_colors = [
        "green" if edge in matching else "red" if edge in new_edges else "black"
        for edge in g.edges
    ]
    colors = ["red" if node[0] == "s" else "blue" for node in g.nodes]
    nx.draw(
        g,
        pos=pos,
        with_labels=True,
        node_color=colors,
        edge_color=edge_colors,
        node_size=150,
    )
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=nx.get_edge_attributes(g, "weight")
    )
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()


def plot_predictions(
    prop_model: DiffusionModel,
    node_values: np.ndarray,
    title: str,
):
    """
    Plots the initial and the current graph with their infection status.
    Additionally, a graph is plotted with the colors of the nodes representing the predicted likelihood of beeing source.
    :param prop_model: propagation model
    :param node_values: an array of values for each node, representing some kind of prediction (i.e. likelihood of beeing source or distance to source)
    """
    g = nx.Graph()
    g.add_nodes_from(prop_model.graph.nodes)
    g.add_edges_from(prop_model.graph.edges)

    # extract the maximum shortest path length from a source node to any other node -> used for coloring
    source_nodes = np.where(
        np.fromiter(prop_model.initial_status.values(), dtype=int) == 1
    )[0]
    max_distance_from_source = np.max(
        [
            np.max(
                np.fromiter(
                    nx.single_source_shortest_path_length(g, source_node).values(),
                    dtype=int,
                )
            )
            for source_node in source_nodes
        ]
    )

    sir_cmap = ListedColormap(["blue", "red", "gray"])
    predictions_cmap = LinearSegmentedColormap.from_list("predictions", ["blue", "red"])

    # initial infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(prop_model.initial_status.values(), dtype=int),
        2,
        f"initial_{title}",
        cmap=sir_cmap,
    )
    # current infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(prop_model.status.values(), dtype=int),
        2,
        f"current_{title}",
        cmap=sir_cmap,
    )
    # predicted initial infection graph
    print(max_distance_from_source)
    plot_graph_with_colors(
        g,
        node_values,
        max_distance_from_source,
        f"prediction_{title}",
        cmap=predictions_cmap,
    )

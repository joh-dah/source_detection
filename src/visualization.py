""" Visualize graphs with the associated predictions. """
from typing import Union
import glob
import os
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from pathlib import Path
import src.constants as const
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm
from src import utils
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
from src.data_processing import SDDataset, process_gcnr_data, process_gcnsi_data


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


def plot_matching_graph(
    g: nx.Graph, matching: list, new_edges: list, title: str = "matching_graph"
):
    """
    Plots the matching graph to debug the min-matching distance metric.
    """
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


def plot_roc_curve(
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    model_name: str,
    dataset_name: str,
):
    """
    Plot ROC curves.
    :param false_positives: the false positives rates
    :param true_positives: the true positives rates
    :param model_name: the name of the model that is evaluated (used for saving the plot)
    :param dataset_name: the name of the dataset the model is evaluated on (used for saving the plot)
    """
    print("Visualize ROC curve:")
    (Path(const.ROC_PATH) / model_name).mkdir(parents=True, exist_ok=True)
    plt.plot(false_positives, true_positives)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(Path(const.ROC_PATH) / model_name / f"{dataset_name}_roc.png")
    plt.close()


def main():
    """
    Visualize graphs with the associated predictions.
    """
    n_graphs = 5
    model_name = utils.latest_model_name()

    if const.MODEL == "GCNR":
        model = GCNR()
    elif const.MODEL == "GCNSI":
        model = GCNSI()

    model = utils.load_model(model, os.path.join(const.MODEL_PATH, f"{model_name}.pth"))
    processed_val_data = utils.load_processed_data("synthetic", True)[:n_graphs]
    raw_val_data = utils.load_raw_data("synthetic", True)[:n_graphs]

    print("Visualize example predictions:")
    for i, data in tqdm(enumerate(raw_val_data)):
        initial_status = data.y
        status = data.x
        edge_index = data.edge_index

        g = nx.Graph()
        g.add_nodes_from(range(len(initial_status)))
        g.add_edges_from(edge_index.t().tolist())

        sir_cmap = ListedColormap(["blue", "red", "gray"])
        predictions_cmap = LinearSegmentedColormap.from_list(
            "predictions", ["red", "blue"]
        )

        # initial infection graph
        plot_graph_with_colors(
            g,
            np.fromiter(initial_status, dtype=int),
            2,
            f"initial_{i}",
            cmap=sir_cmap,
        )

        # current infection graph
        plot_graph_with_colors(
            g,
            np.fromiter(status, dtype=int),
            2,
            f"current_{i}",
            cmap=sir_cmap,
        )

        pred = model(processed_val_data[i].x, processed_val_data[i].edge_index)

        if const.MODEL == "GCNSI":
            # color the 5 highest predictions
            pred = utils.ranked_source_predictions(pred)
            n_colors = 5

        elif const.MODEL == "GCNR":
            # for every node, colorcode the distance to the source. If distance is bigger than 5, color is blue
            n_colors = 4

        # predicted graph
        plot_graph_with_colors(
            g,
            np.fromiter(pred, dtype=int),
            n_colors,
            f"prediction_{i}",
            cmap=predictions_cmap,
        )


if __name__ == "__main__":
    main()

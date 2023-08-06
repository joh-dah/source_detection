""" Visualize graphs with the associated predictions. """
import argparse
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

    plt.figure(figsize=(8, 8))

    node_values = np.clip(node_values, 0, max_colored_value)
    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_color=node_values,
        node_size=200,
        cmap=cmap,
        vmin=0,
        # labels={i: node_values[i] for i in range(len(node_values))},
        vmax=max_colored_value,
        linewidths=[5 if node["source"] == 1 else 1 for node in g.nodes.values()],
        # with_labels=True,
    )
    nodes.set_edgecolor("black")
    nx.draw_networkx_edges(g, pos)
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

    plt.figure(figsize=(20, 20))
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
    thresholds: np.ndarray,
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
    plt.scatter(
        false_positives,
        true_positives,
        c=thresholds,
        cmap="viridis",
        label="ROC curve",
    )
    plt.plot(
        false_positives,
        true_positives,
        color="black",
        linestyle="-",
        alpha=0.5,
    )
    plt.colorbar(label="Threshold")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve for {model_name} on {dataset_name}")
    plt.legend()
    plt.savefig(Path(const.ROC_PATH) / model_name / f"{dataset_name}_roc.png")
    plt.close()


def main():
    """
    Visualize graphs with the associated predictions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="synthetic", help="name of the dataset"
    )
    args = parser.parse_args()
    n_graphs = 10
    model_name = (
        utils.latest_model_name() if const.MODEL_NAME is None else const.MODEL_NAME
    )

    if const.MODEL == "GCNR":
        model = GCNR()
    elif const.MODEL == "GCNSI":
        model = GCNSI()

    model = utils.load_model(model, os.path.join(const.MODEL_PATH, f"{model_name}.pth"))
    processed_val_data = utils.load_processed_data(args.dataset, True)[:n_graphs]
    raw_val_data = utils.load_raw_data(args.dataset, True)[:n_graphs]

    print("Visualize example predictions:")
    for i, data in tqdm(enumerate(raw_val_data)):
        initial_status = data.y.numpy()
        status = data.x
        edge_index = data.edge_index

        g = nx.Graph()
        g.add_nodes_from(range(len(initial_status)))
        nx.set_node_attributes(g, dict(enumerate(initial_status)), "source")
        g.add_edges_from(edge_index.t().tolist())

        sir_cmap = ListedColormap(["blue", "red", "gray"])

        # initial infection graph
        plot_graph_with_colors(
            g,
            np.fromiter(initial_status, dtype=int),
            2,
            f"{args.dataset}_initial_{i}",
            cmap=sir_cmap,
        )

        # current infection graph
        plot_graph_with_colors(
            g,
            np.fromiter(status, dtype=int),
            2,
            f"{args.dataset}_current_{i}",
            cmap=sir_cmap,
        )

        pred = model(processed_val_data[i])

        if const.MODEL == "GCNSI":
            # color the 5 highest predictions
            pred = torch.sigmoid(pred)
            pred = torch.round(pred)
            n_colors = 1
            predictions_cmap = LinearSegmentedColormap.from_list(
                "predictions", ["blue", "red"]
            )

        elif const.MODEL == "GCNR":
            # for every node, colorcode the distance to the source. If distance is bigger than 5, color is blue
            n_colors = 1
            predictions_cmap = LinearSegmentedColormap.from_list(
                "predictions", ["red", "blue"]
            )

        # predicted graph
        plot_graph_with_colors(
            g,
            np.fromiter(pred, dtype=float),
            n_colors,
            f"{args.dataset}_prediction_{i}",
            cmap=predictions_cmap,
        )


if __name__ == "__main__":
    main()

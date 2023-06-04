"""utility functions for data loading and machine learning"""
from os import listdir
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import src.constants as const
import src.vizualization as viz
import networkx as nx
from ndlib.models.DiffusionModel import DiffusionModel
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def load_data(path):
    """Loads data from path."""
    data = []
    print("Load Data:")
    for file in tqdm(listdir(path)):
        data.append(pickle.load(open(path + "/" + file, "rb")))
    return data


def extract_sources(prob_model) -> np.ndarray:
    """Extracts sources from propagation model."""
    initial_values = np.array(list(prob_model.initial_status.values()))
    return np.where(initial_values == 1)[0]


def get_ranked_source_predictions(predictions, n=None):
    """
    Return nodes ranked by predicted probability of beeing source. Selects the n nodes with the highest probability.
    :param predictions: list of tuple predictions of nodes beeing source.
    :param n: amount of nodes to return.
    The second value of the tuple is the probability of the node beeing source.
    :return: list of nodes ranked by predicted probability of beeing source.
    """
    if n is None:
        n = predictions.shape[0]
    source_prob = predictions[:, 1].flatten()
    return torch.topk(source_prob, n).indices


def one_hot_encode(value_list, n_diff_features=-1):
    """
    One-Hot-Encode list of values.
    :param value_list: list of values
    :param n_diff_fearures: amount of different features in list
    :return list of one-hot-encoded values
    """
    label_tensor = torch.tensor(value_list)
    return torch.nn.functional.one_hot(label_tensor, n_diff_features).float()


def save_model(model, name):
    """
    Saves model state to path.
    :param model: model with state
    :param name: name of model
    """
    Path(const.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{const.MODEL_PATH}/{name}.pth")


def load_model(model, path):
    """
    Loads model state from path.
    :param model: model
    :param path: path to model
    :return: model with loaded state
    """
    model.load_state_dict(torch.load(path))
    return model


def vizualize_results(
    model: torch.nn.Module,
    propagation_models: list[DiffusionModel],
    prep_dataset: list[list[Data, torch.Tensor, torch.Tensor]],
):
    """
    Vizualizes the predictions of the model.
    :param model: The model on which predictions are made.
    :param data_set: The data set to vizualize on.
    Contains the graph structure, the features and the labels.
    """
    print("Vizualize Results:")
    for i, propagation_model in tqdm(enumerate(propagation_models)):
        graph_structure, features, _ = prep_dataset[i]
        predictions = model(features, graph_structure.edge_index)
        ranked_predictions = get_ranked_source_predictions(predictions)
        viz.plot_predictions(propagation_model, ranked_predictions, 7, title=f"_{i}")


def vizualize_results_gcnr(
    model: torch.nn.Module,
    propagation_models: list[DiffusionModel],
    prep_dataset: list[list[Data, torch.Tensor, torch.Tensor]],
):
    """
    Vizualizes the predictions of the model.
    :param model: The model on which predictions are made.
    :param data_set: The data set to vizualize on.
    Contains the graph structure, the features and the labels.
    """
    print("Vizualize Results:")
    for i, propagation_model in tqdm(enumerate(propagation_models)):
        graph_structure, features, _ = prep_dataset[i]
        predictions = model(features, graph_structure.edge_index)

        # convert predictions to np array of rounded integer values
        predictions = np.round(predictions.detach().numpy().flatten()).astype(int)

        # extract the maximum shortest path length from a source node to any other node -> used for coloring
        source_nodes = np.where(
            np.fromiter(propagation_model.initial_status.values(), dtype=int) == 1
        )[0]
        g = to_networkx(graph_structure)
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

        viz.plot_predictions(
            propagation_model, predictions, max_distance_from_source, title=f"_{i}"
        )

"""utility functions for data loading and machine learning"""
from os import listdir
import pickle
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import src.constants as const
import src.vizualization as viz


def load_data(path):
    """Loads data from path."""
    data = []
    print("Load Data:")
    for file in tqdm(listdir(path)):
        data.append(pickle.load(open(path + "/" + file, "rb")))
    return data


def get_ranked_source_predictions(predictions, n = None):
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


def vizualize_results(model, raw_dataset, prep_dataset):
    """
    Vizualizes the predictions of the model.
    :param model: The model on which predictions are made.
    :param data_set: The data set to vizualize on.
    Contains the graph structure, the features and the labels.
    """
    print("Vizualize Results:")
    for i, raw_data in tqdm(enumerate(raw_dataset)):
        graph_structure, features, _ = prep_dataset[i]
        predictions = model(features, graph_structure.edge_index)
        ranked_predictions = get_ranked_source_predictions(predictions)
        viz.plot_predictions(raw_data, ranked_predictions, title=f"_{i}")

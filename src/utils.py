"""utility functions for data loading and machine learning"""
from pathlib import Path
import torch
import src.constants as const


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


def get_ranked_source_predictions(predictions, n=None):
    """
    Return nodes ranked by predicted probability of beeing source. Selects the n nodes with the highest probability.
    :param predictions: list of predictions of nodes beeing source.
    :param n: amount of nodes to return.
    :return: list of nodes ranked by predicted probability of beeing source.
    """
    if n is None:
        n = predictions.shape[0]
    return torch.topk(predictions.flatten(), n).indices

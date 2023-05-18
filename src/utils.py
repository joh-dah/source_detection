import torch
import src.constants as const
from pathlib import Path
import pickle
from tqdm import tqdm
from os import listdir


def load_data(path):
    """Loads data from path."""
    data = []
    print("Load Data:")
    for file in tqdm(listdir(path)):
        data.append(pickle.load(open(path + "/" + file, "rb")))
    return data


def get_ranked_source_predictions(predictions):
    """
    Return nodes ranked by predicted probability of beeing source.
    :param predictions: list of tuple predictions of nodes beeing source.
    The second value of the tuple is the probability of the node beeing source.
    :return: list of nodes ranked by predicted probability of beeing source.
    """
    source_prob = predictions[:, 1].flatten()
    return torch.topk(source_prob, len(source_prob)).indices


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

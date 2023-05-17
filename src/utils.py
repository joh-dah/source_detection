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


def get_ranked_source_predictions(
    model, features, edge_index
):  # TODO move somewhere/generalize more
    """
    Return nodes ranked by predicted probability of beeing source.
    :param model: model to make predictions on
    :param features: features for predicion
    """
    out, _ = model(features, edge_index)
    return torch.topk(out[:, 1].flatten(), len(out[:, 1].flatten())).indices


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

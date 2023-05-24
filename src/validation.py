""" This file contains the code for validating the model. """

import src.constants as const
import src.GCN_model as gcn
import src.GCNSI_model as gcnsi
import torch
from torch_geometric.utils import to_networkx
import networkx as nx
from src import utils

def predict_source_probailities(model, graph_structure, features):
    prediction = model(features, graph_structure.edge_index)
    return torch.softmax(prediction, 1)

def most_likely_sources(model, graph_structure, features, n_sources = 1):
    prediction = predict_source_probailities(model, graph_structure, features)
    return torch.topk(prediction[:, 1], n_sources, 0).indices

def sources_with_probability(model, graph_structure, features, probability):
    prediction = predict_source_probailities(model, graph_structure, features)
    return torch.where(prediction[:, 1] > probability)[0]

def distance_to_sources(model, graph_structure, features, labels):
    """
    Calculates the mean of the minimal distances from each source to the nearest predicted source.
    Multiple sources could be assigned to the same predicted source here.
    :param model: The model to evaluate.
    :param graph_structure: The graph structure to evaluate on.
    :param features: The one hot encoded node features: [0, 1] for initally infected, [1, 0] for initially not infected.
    :param labels: The labels of each node: [0, 1] for source, [1, 0] for not source.
    :return: The mean of the minimal distances from each source to the nearest predicted source.
    """
    sources = torch.where(labels[:, 0] == 0)[0]
    predicted_sources = most_likely_sources(model, graph_structure, features, sources.shape[0])
    G = to_networkx(graph_structure)
    mean_dist = 0
    for source in sources.tolist():
        min_dist = labels.shape[0]+1
        for pred_source in predicted_sources.tolist():
            dist = nx.shortest_path_length(G, source=pred_source, target=source)
            min_dist = min(min_dist, dist)
        mean_dist += min_dist
    return mean_dist / sources.shape[0]

def main():
    """Initiates the validation of the classifier specified in the constants file."""

    val_data = utils.load_data(const.DATA_PATH + "/validation")

    n_plots = 5
    prep_val_data = None

    if const.MODEL == "GCN":
        model = gcn.GCN()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        prep_val_data = gcn.prepare_data(val_data)

    elif const.MODEL == "GCNSI":
        model = gcnsi.GCNSI()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        prep_val_data = gcnsi.prepare_data(val_data)

    evaluate(model, prep_val_data)
    utils.vizualize_results(model, val_data[:n_plots], prep_val_data[:n_plots])


if __name__ == "__main__":
    main()

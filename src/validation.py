import json
import glob
from pathlib import Path
import os.path
import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve
import rpasdt.algorithm.models as rpasdt_models
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import DiffusionTypeEnum, SourceDetectionAlgorithm
from torch_geometric.utils.convert import from_networkx

import src.data_processing as dp
import src.constants as const
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
import src.constants as const
from src import visualization as vis
from src import utils


def find_closest_sources(matching_graph, unmatched_nodes):
    """
    Find the minimum weight adjacent edge for each unmatched node.
    :param matching_graph: The graph with only the sources, the predicted sources and the distances between them.
    :param unmatched_nodes: Nodes that didn´t match any other node.
    :return: The minimum weight adjacent edge for each unmatched node.
    """
    new_edges = []
    for node in unmatched_nodes:
        min_weight = float("inf")
        min_weight_node = None
        for neighbor in matching_graph.neighbors(node):
            if matching_graph.get_edge_data(node, neighbor)["weight"] < min_weight:
                min_weight = matching_graph.get_edge_data(node, neighbor)["weight"]
                min_weight_node = neighbor
        new_edges += [(node, min_weight_node), (min_weight_node, node)]
    return new_edges


def min_matching_distance(
    edge_index,
    sources,
    predicted_sources,
    title_for_matching_graph="matching_graph",
):
    """
    Calculates the average minimal matching distance between the sources and the predicted sources.
    This Metric tries to match each source to a predicted source while minimizing the sum of the distances between them.
    When |sources| != |predicted_sources|, some nodes of the smaller set will be matched to multiple nodes of the larger set.
    This penelizes the prediction of too many or too few sources.
    To compare the results of different amounts of sources, the result gets normalized by the number of sources.
    :param graph: The graph to evaluate on.
    :param sources: The indices of the sources.
    :param predicted_sources: The indices of the predicted sources.
    :param title_for_matching_graph: The title for the visualization of the matching graph. Mostly for debugging purposes.
    :return: The avg minimal matching distance between the sources and the predicted sources.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(edge_index[0])))
    G.add_edges_from(edge_index.t().tolist())

    # creating a graph with only the sources, the predicted sources and the distances between them
    matching_graph = nx.Graph()
    avg_dists = []
    for source in sources:
        distances = nx.single_source_shortest_path_length(G, source)
        avg_dists.append(np.mean(list(distances.values())))
        new_edges = [
            ("s" + str(source), str(k), v)
            for k, v in distances.items()
            if k in predicted_sources
        ]
        matching_graph.add_weighted_edges_from(new_edges)

    # finding the minimum weight matching
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
        matching_graph
    )
    matching_list = [(u, v) for u, v in matching.items()]

    # finding the unmatched nodes and adding the closest source to the matching
    unmatched_nodes = [v for v in matching_graph.nodes if v not in matching]
    new_edges = find_closest_sources(matching_graph, unmatched_nodes)
    # vis.plot_matching_graph(
    #     matching_graph, matching_list, new_edges, title_for_matching_graph
    # )
    matching_list += new_edges

    # calculating the sum of the weights of the matching
    min_matching_distance = (
        sum([matching_graph.get_edge_data(k, v)["weight"] for k, v in matching_list])
        / 2
    )  # counting each edge twice
    return min_matching_distance / len(sources), avg_dists


def compute_roc_curve(pred_label_set, data_set):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    and the false positive rates and true positive rates
    for the given data set and the predicted labels.
    """
    all_true_labels = []
    all_pred_labels = []
    for i, pred_labels in enumerate(
        tqdm(pred_label_set[:10], desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        all_true_labels += data_set[i].y.tolist()
        all_pred_labels += pred_labels.tolist()

    pos_label = 0 if const.MODEL == "GCNR" else 1
    false_positive, true_positive, _ = roc_curve(
        all_true_labels, all_pred_labels, pos_label=pos_label
    )
    roc_score = roc_auc_score(all_true_labels, all_pred_labels)
    roc_score = 1 - roc_score if const.MODEL == "GCNR" else roc_score
    return roc_score, true_positive, false_positive


def get_predicted_sources(pred_labels, true_sources):
    """Get the predicted sources from the predicted labels."""
    # TODO: currently we fix the number of predicted sources to the number of true sources
    # we should try to predict the number of sources as well or use a threshold
    ranked_predictions = (utils.get_ranked_source_predictions(pred_labels)).tolist()
    return ranked_predictions[: len(true_sources)]


def get_distance_metrics(pred_label_set, data_set):
    """
    Get the average min matching distance and the average distance to the source in general.
    """
    min_matching_dists = []
    dist_to_source = []

    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_labels = data_set[i].y
        true_sources = torch.where(true_labels == 1)[0].tolist()
        pred_sources = get_predicted_sources(pred_labels, true_sources)
        matching_dist, avg_dist = min_matching_distance(
            data_set[i].edge_index, true_sources, pred_sources
        )
        min_matching_dists.append(matching_dist)
        dist_to_source += avg_dist

    return {
        "min matching distance": np.mean(min_matching_dists),
        "avg dist to source": np.mean(dist_to_source),
    }


def get_TP_FP_metrics(pred_label_set: torch.tensor, data_set: dp.SDDataset):
    """ """
    TPs = 0
    FPs = 0
    n_positives = 0
    n_negatives = 0
    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_sources = torch.where(data_set[i].y == 1)[0].tolist()
        pos_label = 0 if const.MODEL == "GCNR" else 1
        pred_sources = torch.where(torch.round(pred_labels) == pos_label)[0].tolist()
        n_TP = len(np.intersect1d(true_sources, pred_sources))
        TPs += n_TP
        FPs += len(pred_sources) - n_TP
        n_positives += len(true_sources)
        n_negatives += len(pred_labels) - len(true_sources)

    return {
        "True positive rate": TPs / n_positives,
        "False positive rate": FPs / n_negatives,
    }


def get_prediction_metrics(pred_label_set: torch.tensor, data_set: dp.SDDataset):
    """
    Get the average rank of the source, the average prediction for the source
    and additional metrics that help to evaluate the prediction.
    """
    source_ranks = []
    predictions_for_source = []
    general_predictions = []

    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_sources = torch.where(data_set[i].y == 1)[0].tolist()
        ranked_predictions = (utils.get_ranked_source_predictions(pred_labels)).tolist()

        for source in true_sources:
            source_ranks.append(ranked_predictions.index(source))
            predictions_for_source += pred_labels[true_sources].tolist()
            general_predictions += pred_labels.flatten().tolist()

    return {
        "avg rank of source": np.mean(source_ranks),
        "mean number of nodes": np.ceil(len(general_predictions) / len(data_set)),
        "avg prediction for source": np.mean(predictions_for_source),
        "avg prediction over all nodes": np.mean(general_predictions),
        "std prediction over all nodes": np.std(general_predictions),
        "min prediction over all nodes": min(general_predictions),
        "max prediction over all nodes": max(general_predictions),
    }


def get_supervised_metrics(pred_label_set, data_set, model_name):
    """
    Evaluation for models, that predict for every node if it is a source or not.
    Prints the average predicted rank of the real source and the average min matching distance for the predicted sources.
    :param model: The model to evaluate.
    :param prep_val_data: The validation data.
    """
    metrics = {}

    print("Evaluating Model ...")
    roc_score, true_positives, false_positives = compute_roc_curve(
        pred_label_set, data_set
    )
    vis.plot_roc_curve(true_positives, false_positives, model_name)

    metrics |= get_prediction_metrics(pred_label_set, data_set)
    metrics |= get_distance_metrics(pred_label_set, data_set)
    metrics |= get_TP_FP_metrics(pred_label_set, data_set)
    metrics |= {"roc score": roc_score}

    for key, value in metrics.items():
        metrics[key] = round(value, 3)
        print(f"{key}: {metrics[key]}")

    return metrics


def get_min_matching_distance_netsleuth(result):
    """Calculate the average min matching distance for the NETSLEUTH results."""
    min_matching_dists = []
    for mm_r in result.raw_results["NETSLEUTH"]:
        data = from_networkx(mm_r.G)
        min_matching_dists.append(
            min_matching_distance(
                data.edge_index, mm_r.real_sources, mm_r.detected_sources
            )[0]
        )
    return np.mean(min_matching_dists)


def create_simulation_config():
    """Create a rpasdt simulation config with the NETSLEUTH source detector."""
    return rpasdt_models.SourceDetectionSimulationConfig(
        source_detectors={
            "NETSLEUTH": rpasdt_models.SourceDetectorSimulationConfig(
                alg=SourceDetectionAlgorithm.NET_SLEUTH,
                config=rpasdt_models.CommunitiesBasedSourceDetectionConfig(),
            )
        },
    )


def get_unsupervised_metrics(val_data):
    print("Evaluating unsupervised methods ...")
    simulation_config = create_simulation_config()

    result = perform_source_detection_simulation(simulation_config, val_data)
    avg_mm_distance = get_min_matching_distance_netsleuth(result)

    metrics = {
        "avg min matching distance of predicted source": avg_mm_distance,
        "True positive rate": result.aggregated_results["NETSLEUTH"].TPR,
        "False positive rate": result.aggregated_results["NETSLEUTH"].FPR,
    }

    for key, value in metrics.items():
        metrics[key] = round(value, 3)
        print(f"unsupervised - {key}: {metrics[key]}")

    return metrics


def get_predictions(model, data_set):
    """Use the model to make predictions on the dataset"""
    predictions = []
    for data in tqdm(
        data_set, desc="make predictions with model", disable=const.ON_CLUSTER
    ):
        features = data.x
        edge_index = data.edge_index
        if const.MODEL == "GCNSI":
            predictions.append(torch.sigmoid(model(features, edge_index)))
        elif const.MODEL == "GCNR":
            predictions.append(model(features, edge_index))
    return predictions


def main():
    """Initiates the validation of the classifier specified in the constants file."""

    model_name = utils.get_latest_model_name()

    if const.MODEL == "GCNR":
        model = GCNR()
    elif const.MODEL == "GCNSI":
        model = GCNSI()

    model = utils.load_model(model, os.path.join(const.MODEL_PATH, f"{model_name}.pth"))
    processed_val_data = utils.load_processed_data("validation")
    raw_val_data = utils.load_raw_data("validation")
    pred_labels = get_predictions(model, processed_val_data)

    metrics_dict = {}
    metrics_dict["supervised"] = get_supervised_metrics(
        pred_labels, raw_val_data, model_name
    )
    metrics_dict["unsupervised"] = get_unsupervised_metrics(raw_val_data)
    metrics_dict["parameters"] = json.load(open("params.json"))
    utils.save_metrics(metrics_dict, model_name)


if __name__ == "__main__":
    main()

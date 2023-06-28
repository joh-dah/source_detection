""" Initiates the validation of the classifier specified in the constants file. """
import argparse
import yaml
import os.path
import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve
import rpasdt.algorithm.models as rpasdt_models
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import SourceDetectionAlgorithm
from torch_geometric.utils.convert import from_networkx

import src.data_processing as dp
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
import src.constants as const
from src import visualization as vis
from src import utils


def find_closest_sources(matching_graph: nx.Graph, unmatched_nodes: list) -> list:
    """
    Find the minimum weight adjacent edge for each unmatched node.
    :param matching_graph: the graph with only the sources, the predicted sources and the distances between them
    :param unmatched_nodes: nodes that didn´t match any other node
    :return: list of the minimum weight adjacent edge for each unmatched node
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
    edge_index: torch.tensor, sources: list, predicted_sources: list
) -> tuple[float, list]:
    """
    Calculates the average minimal matching distance between the sources and the predicted sources.
    This Metric tries to match each source to a predicted source while minimizing the sum of the distances between them.
    When |sources| != |predicted_sources|, some nodes of the smaller set will be matched to multiple nodes of the larger set.
    This penelizes the prediction of too many or too few sources.
    To compare the results of different amounts of sources, the result gets normalized by the number of sources.
    :param edge_index: The edge_index of the graph to evaluate on.
    :param sources: The indices of the sources.
    :param predicted_sources: The indices of the predicted sources.
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


def compute_roc_curve(
    pred_label_set: list, data_set: list
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) and the false positive rates and
    true positive rates for the given data set and the predicted labels.
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param data_set: list of data instances containing true labels
    :return: ROC AUC score, true positive rates, and false positive rates
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


def predicted_sources(pred_labels: list, true_sources: list) -> list:
    """
    Get the predicted sources from the predicted labels.
    :param pred_labels: the predicted labels for the instances
    :param true_sources: list of true sources
    :return: the predicted sources based on the predicted labels
    """
    # TODO: currently we fix the number of predicted sources to the number of true sources
    # we should try to predict the number of sources as well or use a threshold
    ranked_predictions = (utils.ranked_source_predictions(pred_labels)).tolist()
    return ranked_predictions[: len(true_sources)]


def distance_metrics(pred_label_set: list, data_set: list) -> dict:
    """
    Get the average min matching distance and the average distance to the source in general.
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param data_set: list of data instances containing true labels
    :return: dictionary with the average minimum matching distance and average distance to the source
    """
    min_matching_dists = []
    dist_to_source = []

    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_labels = data_set[i].y
        true_sources = torch.where(true_labels == 1)[0].tolist()
        pred_sources = predicted_sources(pred_labels, true_sources)
        matching_dist, avg_dist = min_matching_distance(
            data_set[i].edge_index, true_sources, pred_sources
        )
        min_matching_dists.append(matching_dist)
        dist_to_source += avg_dist

    return {
        "min matching distance": np.mean(min_matching_dists),
        "avg dist to source": np.mean(dist_to_source),
    }


def TP_FP_metrics(pred_label_set: list, data_set: list) -> dict:
    """
    Calculate the true positive rate and false positive rate metrics based on the predicted labels and data set.
    :param pred_label_set: predicted labels for each data instance in the data set
    :param data_set: a data set containing true labels
    :return: dictionary with the true positive rate and false positive rate.
    """
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


def prediction_metrics(pred_label_set: list, data_set: list) -> dict:
    """
    Get the average rank of the source, the average prediction for the source
    and additional metrics that help to evaluate the prediction.
    :param pred_label_set: predicted labels for each data instance in the data set
    :param data_set: a data set containing true labels
    :return: dictionary with prediction metrics
    """
    source_ranks = []
    predictions_for_source = []
    general_predictions = []

    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_sources = torch.where(data_set[i].y == 1)[0].tolist()
        ranked_predictions = (utils.ranked_source_predictions(pred_labels)).tolist()

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


def supervised_metrics(pred_label_set: list, data_set: list, model_name: str) -> dict:
    """
    Performs supervised evaluation metrics for models that predict whether each node is a source or not.
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param data_set: the data set containing true labels
    :param model_name: name of the model being evaluated
    :return: dictionary containing the evaluation metrics
    """
    metrics = {}

    print("Evaluating Model ...")
    roc_score, true_positives, false_positives = compute_roc_curve(
        pred_label_set, data_set
    )
    vis.plot_roc_curve(true_positives, false_positives, model_name)

    metrics |= prediction_metrics(pred_label_set, data_set)
    metrics |= distance_metrics(pred_label_set, data_set)
    metrics |= TP_FP_metrics(pred_label_set, data_set)
    metrics |= {"roc score": roc_score}

    for key, value in metrics.items():
        metrics[key] = round(value, 3)
        print(f"{key}: {metrics[key]}")

    return metrics


def min_matching_distance_netsleuth(
    result: rpasdt_models.SourceDetectionSimulationResult,
) -> float:
    """
    Calculate the average minimum matching distance for the NETSLEUTH results.
    :param result: NETSLEUTH results
    :return: average minimum matching distance
    """
    min_matching_dists = []
    for mm_r in result.raw_results["NETSLEUTH"]:
        data = from_networkx(mm_r.G)
        min_matching_dists.append(
            min_matching_distance(
                data.edge_index, mm_r.real_sources, mm_r.detected_sources
            )[0]
        )
    return np.mean(min_matching_dists)


def create_simulation_config() -> rpasdt_models.SourceDetectionSimulationConfig:
    """
    Create a rpasdt simulation config with the NETSLEUTH source detector.
    """
    return rpasdt_models.SourceDetectionSimulationConfig(
        source_detectors={
            "NETSLEUTH": rpasdt_models.SourceDetectorSimulationConfig(
                alg=SourceDetectionAlgorithm.NET_SLEUTH,
                config=rpasdt_models.CommunitiesBasedSourceDetectionConfig(),
            )
        }
    )


def unsupervised_metrics(val_data: list) -> dict:
    """
    Performs unsupervised evaluation metrics for models.
    :param val_data: the validation data
    :return: dictionary containing the evaluation metrics
    """
    print("Evaluating unsupervised methods ...")
    simulation_config = create_simulation_config()

    result = perform_source_detection_simulation(simulation_config, val_data)
    avg_mm_distance = min_matching_distance_netsleuth(result)

    metrics = {
        "avg min matching distance of predicted source": avg_mm_distance,
        "True positive rate": result.aggregated_results["NETSLEUTH"].TPR,
        "False positive rate": result.aggregated_results["NETSLEUTH"].FPR,
    }

    for key, value in metrics.items():
        metrics[key] = round(value, 3)
        print(f"unsupervised - {key}: {metrics[key]}")

    return metrics


def data_stats(raw_data_set: list) -> dict:
    """
    Calculates various graph-related statistics and infection-related statistics for the provided raw data set.
    :param raw_data_set: the raw data set.
    :return: dictionary containing the calculated statistics
    """
    n_nodes = []
    n_sources = []
    centrality = []
    n_nodes_infected = []
    precent_infected = []
    for data in tqdm(raw_data_set, desc="get data stats"):
        n_nodes.append(len(data.y))
        n_sources.append(len(torch.where(data.y == 1)[0].tolist()))
        n_nodes_infected.append(len(torch.where(data.x == 1)[0].tolist()))
        precent_infected.append(n_nodes_infected[-1] / n_nodes[-1])
        graph = nx.from_edgelist(data.edge_index.t().tolist())
        centrality.append(np.mean(list(nx.degree_centrality(graph).values())))

    stats = {
        "graph stats": {
            "avg number of nodes": np.mean(n_nodes),
            "avg centrality": np.mean(centrality),
            "std centrality": np.std(centrality),
        },
        "infection stats": {
            "avg number of sources": np.mean(n_sources),
            "avg number of infected nodes": np.mean(n_nodes_infected),
            "std number of infected nodes": np.std(n_nodes_infected),
            "avg percent infected nodes": np.mean(precent_infected),
            "std percent infected nodes": np.std(precent_infected),
        },
    }

    for key, value in stats.items():
        for k, v in value.items():
            stats[key][k] = round(v, 3)

    return stats


def predictions(model: torch.nn.Module, data_set: dp.SDDataset) -> list:
    """
    Generate predictions using the specified model for the given data set.
    :param model: the model used for making predictions
    :param data_set: the data set to make predictions on
    :return: predictions generated by the model
    """
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
    """
    Initiates the validation of the classifier specified in the constants file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="synthetic", help="name of the dataset"
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    model_name = utils.latest_model_name()

    if const.MODEL == "GCNR":
        model = GCNR()
    elif const.MODEL == "GCNSI":
        model = GCNSI()

    model = utils.load_model(model, os.path.join(const.MODEL_PATH, f"{model_name}.pth"))
    processed_val_data = utils.load_processed_data(dataset, True)
    raw_val_data = utils.load_raw_data(dataset, True)
    pred_labels = predictions(model, processed_val_data)

    metrics_dict = {}
    metrics_dict["dataset"] = dataset
    metrics_dict["supervised"] = supervised_metrics(
        pred_labels, raw_val_data, model_name
    )
    metrics_dict["unsupervised"] = unsupervised_metrics(raw_val_data)
    metrics_dict["data stats"] = data_stats(raw_val_data)
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))
    utils.save_metrics(metrics_dict, model_name, dataset)


if __name__ == "__main__":
    main()

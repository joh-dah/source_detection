import json
import numpy as np
import glob
import os.path
import torch
from tqdm import tqdm
import networkx as nx
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
import src.constants as const
from src import visualization as vis
from src import utils
from src.data_processing import SDDataset, process_gcnr_data, process_gcnsi_data


def predict_source_probailities(model, graph_structure, features):
    """
    Prediction about the probability that nodes are sources.
    :param model: The model to evaluate.
    :param graph_structure: The graph structure.
    :param features: The features of the nodes.
    :return: Probabilities that nodes are sources.
    """
    prediction = model(features, graph_structure.edge_index)
    return torch.softmax(prediction, 1)


def threshold_predicted_sources(model, graph_structure, features, threshold):
    """
    Find nodes whose probability of being a source is higher than the threshold.
    :param model: The model to evaluate.
    :param graph_structure: The graph structure.
    :param features: The features of the nodes.
    :param threshold: Value that tells how high the probability must be to assume that the node is a source.
    :return: Nodes whose probability of being a source is higher than the threshold.
    """
    prediction = predict_source_probailities(model, graph_structure, features)
    return torch.where(prediction[:, 1] > threshold)[0]


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
    for source in sources:
        distances = nx.single_source_shortest_path_length(G, source)
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
    return min_matching_distance / len(sources)


def compute_roc_curve(predictions, labels):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores, false positive rates and true positive rates.
    :param predictions: Predicted value for a node to be a source.
    :param labels: Actual sources.
    :return: Area under the roc curve, false positive rates and true positive rates.
    """
    source_prob = predictions.flatten()
    false_positive, true_positive, thresholds = roc_curve(
        labels.tolist(), source_prob.tolist()
    )
    roc_score = roc_auc_score(labels.tolist(), source_prob.tolist())
    return roc_score, false_positive, true_positive


def evaluate_source_predictions(model, val_data):
    """
    Evaluation for models, that predict for every node if it is a source or not.
    Prints the average predicted rank of the real source and the average min matching distance for the predicted sources.
    :param model: The model to evaluate.
    :param prep_val_data: The validation data.
    """
    min_matching_distances = []
    ranks = []
    roc_scores = []
    false_positives = []
    true_positives = []
    n_plots = 5
    for i, data in enumerate(tqdm(val_data, desc="evaluate model")):
        labels = data.y
        features = data.x
        edge_index = data.edge_index
        sources = torch.where(labels == 1)[0]
        predictions = model(features, edge_index)
        ranked_predictions = (utils.get_ranked_source_predictions(predictions)).tolist()
        for source in sources:
            ranks.append(ranked_predictions.index(source))
        top_n_predictions = utils.get_ranked_source_predictions(
            predictions, len(sources)
        )
        # currently we are fixing the number of predicted sources to the number of sources in the graph
        min_matching_distances.append(
            min_matching_distance(edge_index, sources.tolist(), top_n_predictions.tolist())
        )

        roc_score, false_positive, true_positive = compute_roc_curve(
            predictions, labels
        )
        roc_scores.append(roc_score)
        false_positives.append(false_positive)
        true_positives.append(true_positive)

    metrics_dict = {
        "avg predicted rank of source": np.mean(ranks),
        "avg min matching distance of predicted source": np.mean(
            min_matching_distances
        ),
        "avg roc score": round(sum(roc_scores) / len(roc_scores), 2),
    }

    vis.plot_roc_curve(false_positives[:n_plots], true_positives[:n_plots])

    return metrics_dict


def evaluate_source_distance(model, val_data):
    """
    Evaluates the model on the validation data.
    :param model: The model to evaluate.
    :param prep_val_data: The validation data.
    """
    pred_source_distances = []
    pred_distances = []
    for data in tqdm(val_data, desc="evaluate model"):
        labels = data.y
        features = data.x
        edge_index = data.edge_index
        sources = torch.where(labels == 0)[0]
        predictions = model(features, edge_index)
        pred_distances += predictions.tolist()
        pred_source_distances += predictions[sources].tolist()

    metrics_dict = {
        "predicted source_distance of sources:": np.mean(pred_source_distances),
        "avg predicted source_distances": np.mean(pred_distances),
    }

    return metrics_dict


def main():
    """Initiates the validation of the classifier specified in the constants file."""

    model_files = glob.glob(const.MODEL_PATH + r"/*[0-9].pth")
    last_model_file = max(model_files, key=os.path.getctime)
    model_name = last_model_file.split("/")[-1].split(".")[0]
    print(f"loading model: {last_model_file}")

    if const.MODEL == "GCNSI":
        model = GCNSI()
        model = utils.load_model(model, last_model_file)
        val_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnsi_data)[
            const.TRAINING_SIZE :
        ]
        metrics_dict = evaluate_source_predictions(model, val_data)

    elif const.MODEL == "GCNR":
        model = GCNR()
        model = utils.load_model(model, last_model_file)
        val_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnr_data)[
            const.TRAINING_SIZE :
        ]
        metrics_dict = evaluate_source_distance(model, val_data)

    for key, value in metrics_dict.items():
        print(f"{key}: {value}")

    Path(const.REPORT_PATH).mkdir(parents=True, exist_ok=True)
    json.dump(
        metrics_dict,
        open(f"{const.REPORT_PATH}/{model_name}.json", "w"),
        indent=4,
    )
    json.dump(
        metrics_dict,
        open(f"{const.REPORT_PATH}/latest.json", "w"),
        indent=4,
    )


if __name__ == "__main__":
    main()

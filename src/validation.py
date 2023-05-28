""" This file contains the code for validating the model. """

import src.constants as const
import src.GCN_model as gcn
import src.GCNSI_model as gcnsi
import src.vizualization as viz
from src import utils

import torch
from tqdm import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve


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
            if matching_graph.get_edge_data(node, neighbor)['weight'] < min_weight:
                min_weight = matching_graph.get_edge_data(node, neighbor)['weight']
                min_weight_node = neighbor
        new_edges += [(node, min_weight_node), (min_weight_node, node)]
    return new_edges

def min_matching_distance(graph, sources, predicted_sources, title_for_matching_graph = "matching_graph"):
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
    G = to_networkx(graph)

    # creating a graph with only the sources, the predicted sources and the distances between them
    matching_graph = nx.Graph()
    for source in sources:
        distances = nx.single_source_shortest_path_length(G, source)
        new_edges = [("s" + str(source) ,str(k),v) for k, v in distances.items() if k in predicted_sources.tolist()]
        matching_graph.add_weighted_edges_from(new_edges)

    # finding the minimum weight matching
    matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(matching_graph)
    matching_list = [(u,v) for u,v in matching.items()]

    # finding the unmatched nodes and adding the closest source to the matching
    unmatched_nodes = [v for v in matching_graph.nodes if v not in matching]
    new_edges = find_closest_sources(matching_graph, unmatched_nodes)
    viz.plot_matching_graph(matching_graph, matching_list, new_edges, title_for_matching_graph)
    matching_list += new_edges
    
    # calculating the sum of the weights of the matching
    min_matching_distance = sum([matching_graph.get_edge_data(k, v)['weight'] for k, v in matching_list]) / 2 # counting each edge twice
    return min_matching_distance / len(sources)


def compute_roc_curve(predictions, labels):
    '''
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores, false positive rates and true positive rates.
    :param predictions: Predicted value for a node to be a source.
    :param labels: Actual sources.
    :return: Area under the roc curve, false positive rates and true positive rates.
    '''
    soft_max_predictions = torch.softmax(predictions, 1)
    source_prob = soft_max_predictions[:, 1].flatten()
    false_positive, true_positive, thresholds = roc_curve(labels[:, 1].tolist(), source_prob.tolist())
    roc_score = roc_auc_score(labels[:, 1].tolist(), source_prob.tolist())
    return roc_score, false_positive, true_positive


def evaluate(model, prep_val_data):
    """
    Evaluates the model on the validation data. 
    :param model: The model to evaluate.
    :param prep_val_data: The validation data.
    """
    min_matching_distances = []
    ranks = []
    roc_scores = []
    false_positives = []
    true_positives = []
    n_plots = 5

    for i, (graph_structure, features, labels) in enumerate(tqdm(prep_val_data, desc="evaluate model")):
        sources = (torch.where(labels[:, 0] == 0)[0]).tolist()
        predictions = model(features, graph_structure.edge_index)
        ranked_predictions = (utils.get_ranked_source_predictions(predictions)).tolist()
        for source in sources:
            ranks.append(ranked_predictions.index(source))
        top_n_predictions = utils.get_ranked_source_predictions(predictions, len(sources))

        # currently we are fixing the number of predicted sources to the number of sources in the graph
        min_matching_distances.append(min_matching_distance(graph_structure, sources, top_n_predictions))

        roc_score, false_positive, true_positive = compute_roc_curve(predictions, labels)
        roc_scores.append(roc_score)
        false_positives.append(false_positive)
        true_positives.append(true_positive)

    print(f"Average rank of predicted source: {sum(ranks)/len(ranks)}")
    print(f"Average min matching distance: {sum(min_matching_distances)/len(min_matching_distances)}")
    print(f"Average roc score: {round(sum(roc_scores)/len(roc_scores), 2)}")
    viz.plot_roc_curve(false_positives[:n_plots], true_positives[:n_plots])


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

import src.constants as const
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
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
    edge_index, sources, predicted_sources, title_for_matching_graph="matching_graph"
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
    for source in sources.tolist():
        distances = nx.single_source_shortest_path_length(G, source)
        new_edges = [
            ("s" + str(source), str(k), v)
            for k, v in distances.items()
            if k in predicted_sources.tolist()
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


def evaluate_source_predictions(model, val_data):
    """
    Evaluation for models, that predict for every node if it is a source or not.
    Prints the average predicted rank of the real source and the average min matching distance for the predicted sources.
    :param model: The model to evaluate.
    :param prep_val_data: The validation data.
    """
    min_matching_distances = []
    ranks = []
    for i, data in enumerate(tqdm(val_data, desc="evaluate model")):
        labels = data.y
        features = data.x
        edge_index = data.edge_index
        sources = torch.where(labels[:, 0] == 0)[0]
        predictions = model(features, edge_index)
        ranked_predictions = (utils.get_ranked_source_predictions(predictions)).tolist()
        for source in sources.tolist():
            ranks.append(ranked_predictions.index(source))
        top_n_predictions = utils.get_ranked_source_predictions(
            predictions, len(sources)
        )
        # currently we are fixing the number of predicted sources to the number of sources in the graph
        min_matching_distances.append(
            min_matching_distance(edge_index, sources, top_n_predictions)
        )

    print(f"Average predicted rank of source: {np.mean(ranks)}")
    print(
        f"Average min matching distance of predicted source: {np.mean(min_matching_distances)}"
    )


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

    print(
        f"Average predicted source_distance for the sources: {np.mean(pred_source_distances)}"
    )
    print(f"Average predicted distance: {np.mean(pred_distances)}")


def main():
    """Initiates the validation of the classifier specified in the constants file."""

    if const.MODEL == "GCNSI":
        model = GCNSI()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        val_data = SDDataset(const.DATA_PATH, transform=process_gcnr_data)[
            const.TRAINING_SIZE :
        ]
        evaluate_source_predictions(model, val_data)

    elif const.MODEL == "GCNR":
        model = GCNR()
        model = utils.load_model(model, f"{const.MODEL_PATH}/{const.MODEL}_latest.pth")
        val_data = SDDataset(const.DATA_PATH, transform=process_gcnsi_data)[
            const.TRAINING_SIZE :
        ]
        evaluate_source_distance(model, val_data)


if __name__ == "__main__":
    main()

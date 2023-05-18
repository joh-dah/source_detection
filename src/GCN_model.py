import torch
from pathlib import Path
from torch_geometric.nn import GCNConv
import src.constants as const
import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx, to_networkx
import src.utils as utils
from tqdm import tqdm
import src.vizualization as viz


class GCN(torch.nn.Module):
    """Graph Convolutional Network."""

    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(const.N_FEATURES, 3)
        self.conv2 = GCNConv(3, 6)
        self.conv3 = GCNConv(6, 3)
        self.conv4 = GCNConv(3, 3)
        self.conv5 = GCNConv(3, 3)
        self.conv6 = GCNConv(3, 2)
        self.conv7 = GCNConv(2, 2)
        self.classifier = torch.nn.Linear(2, 2)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        h = self.conv4(h, edge_index)
        h = h.tanh()
        h = self.conv5(h, edge_index)
        h = h.tanh()
        h = self.conv6(h, edge_index)
        h = h.tanh()
        h = self.conv7(h, edge_index)
        out = self.classifier(h)
        return out, h


def train_single_epoch(model, graph, features, labels):
    """
    Trains the model for one epoch.
    :param model: The model to train.
    :param graph: The graph structure.
    :param features: The features of the nodes.
    :param labels: The labels of the nodes.
    :return: The loss and the embeddings.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=const.LEARNING_RATE)

    optimizer.zero_grad()
    out, h = model(features, graph.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    return loss, h


def train(model, data):
    """
    Trains the model.
    :param model: The model to train.
    :param data: The data to train on. Contains the graph structure, the features and the labels.
    :return: The trained model.
    """
    epochs = range(1, const.EPOCHS)
    losses = []
    embeddings = []
    print("Train Model:")
    for epoch in tqdm(epochs):
        running_loss = 0.0
        for graph_structure, features, labels in data:
            loss, h = train_single_epoch(model, graph_structure, features, labels)
            running_loss += loss
            losses.append(loss)
            embeddings.append(h)
        print(f"Epoch: {epoch}\tLoss: {running_loss:.4f}")
    return model


def evaluate(model, data_set):
    """
    Evaluates the model.
    :param model: The model to evaluate.
    :param data_set: The data set to evaluate on. Contains the graph structure, the features and the labels.
    """
    ranks = []
    print("Evaluate Model:")
    for graph_structure, features, labels in tqdm(data_set):
        predictions, _ = model(features, graph_structure.edge_index)
        ranked_predictions = utils.get_ranked_source_predictions(predictions)
        source = labels.tolist().index([0, 1])
        ranks.append(ranked_predictions.tolist().index(source))

    print("Average rank of predicted source:")
    print(np.mean(ranks))


def prepare_data(prop_models):
    """
    Prepares the data for the training.
    Extracts the graph structure, the features and the labels from the propagation model.
    :param prop_models: The propagation models to extract the data from.
    :return: List of tuples containing the graph structure, the features and the labels.
    """
    data = []
    print("Prepare Data:")
    for prop_model in tqdm(prop_models):
        G = nx.Graph()
        G.add_nodes_from(prop_model.graph.nodes)
        G.add_edges_from(prop_model.graph.edges)
        graph_structure = from_networkx(G)
        features = utils.one_hot_encode(
            list(prop_model.status.values()), const.N_FEATURES
        )
        labels = utils.one_hot_encode(
            list(prop_model.initial_status.values()), const.N_CLASSES
        )
        data.append([graph_structure, features, labels])
    return data


def vizualize_results(model, data_set):
    """
    Vizualizes the predictions of the model.
    :param model: The model on which predictions are made.
    :param data_set: The data set to vizualize on. Contains the graph structure, the features and the labels.
    """
    print("Vizualize Results:")
    prep_data = prepare_data(data_set)
    for i, raw_data in tqdm(enumerate(data_set)):
        graph_structure, features, _ = prep_data[i]
        predictions, _ = model(features, graph_structure.edge_index)
        ranked_predictions = utils.get_ranked_source_predictions(predictions)
        viz.plot_predictions(raw_data, ranked_predictions, title=f"_{i}")

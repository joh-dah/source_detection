import numpy as np
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm
import src.constants as const
from src import utils


class GCNSI(torch.nn.Module):
    """
    Graph Convolutional Network.
    Based on paper: https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    """

    def __init__(self):
        super(GCNSI, self).__init__()
        torch.manual_seed(42)
        self.conv_first = GCNConv(const.GCNSI_N_FEATURES, const.HIDDEN_SIZE)
        self.conv = GCNConv(const.HIDDEN_SIZE, const.HIDDEN_SIZE)
        self.classifier = torch.nn.Linear(const.HIDDEN_SIZE, const.N_CLASSES)

    def forward(self, x, edge_index):
        h = self.conv_first(x, edge_index)
        h = h.relu()
        for i in range(1, const.LAYERS):
            h = self.conv(h, edge_index)
            h = h.relu()
        out = self.classifier(h)
        return out


def train_single_epoch(model, graph, features, labels):
    """
    Trains the model for one epoch.
    :param model: The model to train.
    :param graph: The graph structure.
    :param features: The features of the nodes.
    :param labels: The labels of the nodes.
    :return: The loss
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)

    optimizer.zero_grad()
    out = model(features, graph.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    return loss


def train(model, data):
    """
    Trains the model.
    :param model: The model to train.
    :param data: The data to train on. Contains the graph structure, the features and the labels.
    :return: The trained model.
    """
    epochs = range(1, const.EPOCHS)
    losses = []
    print("Train Model:")
    for epoch in tqdm(epochs):
        running_loss = 0.0
        for graph_structure, features, labels in data:
            loss = train_single_epoch(model, graph_structure, features, labels)
            running_loss += loss
            losses.append(loss)
        print(f"Epoch: {epoch}\tLoss: {running_loss:.4f}")
    return model


def prepare_input_features(graph, prop_model, a):
    """
    Prepares the input features for the GCNSI model according to the paper:
    https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    :param graph: The graph structure.
    :param model: The propagation model.
    :param a: controls the influence that a node gets from its neighbors
    :return: The stacked input features for the GCNSI model with shape (const.N_NODES, 4)
    """
    Y = np.array(list(prop_model.status.values())).T
    S = nx.normalized_laplacian_matrix(graph)
    V3 = Y.copy()
    Y = [-1 if x == 0 else 1 for x in Y]
    V4 = [-1 if x == -1 else 0 for x in Y]
    I = np.identity(len(Y))
    d1 = Y
    d2 = (1 - a) * np.linalg.inv(I - a * S).dot(Y)
    d3 = (1 - a) * np.linalg.inv(I - a * S).dot(V3)
    d4 = (1 - a) * np.linalg.inv(I - a * S).dot(V4)
    con = np.column_stack((d1, d2, d3, d4))
    return torch.from_numpy(con).float()


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
        features = prepare_input_features(G, prop_model, const.ALPHA)
        labels = utils.one_hot_encode(
            list(prop_model.initial_status.values()), const.N_CLASSES
        )
        data.append([graph_structure, features, labels])
    return data

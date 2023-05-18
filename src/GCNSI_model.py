import numpy as np
import networkx as nx
import src.constants as const
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils.convert import from_networkx, to_networkx
import src.utils as utils

class GCNSI(torch.nn.Module):
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
        return out, h

def train_single_epoch(model, graph, features, labels):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=const.WEIGHT_DECAY)

    optimizer.zero_grad()
    out, h = model(features, graph.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    return loss


def train(model, data):
    epochs = range(1, const.EPOCHS)
    for epoch in epochs:
        running_loss = 0.0
        for graph_structure, features, labels in data:
            loss = train_single_epoch(model, graph_structure, features, labels)
            running_loss += loss
        print(f"Epoch: {epoch}\tLoss: {running_loss:.4f}")
    return model


def evaluate(model, data):
    sources = []
    predictions = []
    ranks = []
    for graph_structure, features, labels in data:
        source = labels.tolist().index([0, 1])
        ranked_predictions = utils.get_ranked_source_predictions(
            model, features, graph_structure.edge_index
        )
        sources.append(source)
        predictions.append(predictions)
        ranks.append(ranked_predictions.tolist().index(source))

    print("Average rank of predicted source:")
    print(np.mean(ranks))

def prepare_input_features(graph, model, a):
    vector = [1]
    vector[0] = list(model.status.values())
    Y = np.array(vector).T
    S = nx.normalized_laplacian_matrix(graph)
    V3 = Y.copy()
    V4 = Y.copy()
    for i in range(0, len(Y)):
        if Y[i] == 0:
            V3[i] = 0
            V4[i] = -1
            Y[i] = -1
        else:
            V3[i] = 1
            V4[i] = 0
            Y[i] = 1
    I = np.identity(len(Y))
    d1 = Y
    d2 = (1-a)*np.linalg.inv(I-a*S).dot(Y)
    d3 = (1-a)*np.linalg.inv(I-a*S).dot(V3)
    d4 = (1-a)*np.linalg.inv(I-a*S).dot(V4)
    con = np.column_stack((d1, d2, d3, d4))
    return torch.from_numpy(con).float()


def prepare_data(models):
    data = []
    for model in models:
        G = nx.Graph()
        G.add_nodes_from(model.graph.nodes)
        G.add_edges_from(model.graph.edges)
        graph_structure = from_networkx(G)
        features = prepare_input_features(G, model, const.LEARNING_RATE)
        labels = utils.one_hot_encode(
            list(model.initial_status.values()), const.N_CLASSES
        )
        data.append([graph_structure, features, labels])
    return data





import torch
from pathlib import Path
from torch_geometric.nn import GCNConv
import constants as const
import numpy as np


class GCN(torch.nn.Module):
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


def save(net, name):
    Path(const.NET_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), f"{const.NET_PATH}/{name}.pth")


def train_single_epoch(net, graph, features, labels):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=const.LEARNING_RATE)

    optimizer.zero_grad()
    out, h = net(features, graph.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    return loss, h


def train(net, data):
    epochs = range(1, const.EPOCHS)
    losses = []
    embeddings = []
    for epoch in epochs:
        running_loss = 0.0
        for graph_structure, features, labels in data:
            loss, h = train_single_epoch(net, graph_structure, features, labels)
            running_loss += loss
            losses.append(loss)
            embeddings.append(h)
        print(f"Epoch: {epoch}\tLoss: {running_loss:.4f}")
    save(net, "gcn")
    return net


def get_ranked_source_predictions(net, features, edge_index):
    """
    return nodes with highest probability of beeing source.
    """
    out, _ = net(features, edge_index)
    return torch.topk(out[:, 1].flatten(), len(out[:, 1].flatten())).indices


def evaluate(net, data):
    sources = []
    predictions = []
    ranks = []
    for graph_structure, features, labels in data:
        source = labels.tolist().index([0, 1])
        ranked_predictions = get_ranked_source_predictions(
            net, features, graph_structure.edge_index
        )
        sources.append(source)
        predictions.append(predictions)
        ranks.append(ranked_predictions.tolist().index(source))

    print("Average rank of predicted source:")
    print(np.mean(ranks))

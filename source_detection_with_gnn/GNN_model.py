import torch
from torch_geometric.nn import GCNConv
import constants as const


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(const.N_FEATURES, 3)
        self.conv2 = GCNConv(3, 6)
        self.conv3 = GCNConv(6, 3)
        self.conv4 = GCNConv(3, 3)
        self.conv5 = GCNConv(3, 3)
        self.conv6 = GCNConv(3, 3)
        self.conv7 = GCNConv(3, 3)
        self.classifier = torch.nn.Linear(3, const.N_CLASSES)

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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    optimizer.zero_grad()
    out, h = model(features, graph.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    return loss, h


def train(model, graph, features, labels):
    epochs = range(1, const.EPOCHS)
    losses = []
    embeddings = []
    for epoch in epochs:
        loss, h = train_single_epoch(model, graph, features, labels)
        losses.append(loss)
        embeddings.append(h)
        print(f"Epoch: {epoch}\tLoss: {loss:.4f}")

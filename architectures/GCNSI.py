import torch
from torch_geometric.nn import GCNConv
import src.constants as const


class GCNSI(torch.nn.Module):
    """
    Graph Convolutional Network.
    Based on paper: https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    """

    def __init__(self):
        super(GCNSI, self).__init__()
        self.conv_first = GCNConv(const.GCNSI_N_FEATURES, const.HIDDEN_SIZE)
        self.conv = GCNConv(const.HIDDEN_SIZE, const.HIDDEN_SIZE)
        self.classifier = torch.nn.Linear(const.HIDDEN_SIZE, 1)

    def forward(self, data):
        h = self.conv_first(data.x, data.edge_index)
        h = h.relu()
        for i in range(1, const.LAYERS):
            h = self.conv(h, data.edge_index)
            h = h.relu()
        out = self.classifier(h)
        return out

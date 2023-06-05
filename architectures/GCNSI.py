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

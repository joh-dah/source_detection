import torch
from torch_geometric.nn import GCNConv
import src.constants as const


class GCNR(torch.nn.Module):
    def __init__(self):
        super(GCNR, self).__init__()
        self.conv_first = GCNConv(const.GCNR_N_FEATURES, const.HIDDEN_SIZE)
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

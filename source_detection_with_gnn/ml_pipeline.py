from GNN_model import GCN, train, evaluate
from torch_geometric.utils.convert import from_networkx
import data_creation as dc
import constants as const
import torch
import utils


def pipeline():
    gcn = GCN()

    train_data = dc.create_data_set(40)
    gcn = train(gcn, train_data)

    val_data = dc.create_data_set(40)
    evaluate(gcn, val_data)

    utils.plot_predictions(gcn, 2)

    # trained_gcn = GCN()
    # trained_gcn.load_state_dict(torch.load(const.NET_PATH + "/gcn.pth"))


if __name__ == "__main__":
    pipeline()

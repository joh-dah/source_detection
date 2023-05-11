from torch_geometric.utils.convert import from_networkx
from GNN_model import GCN, train
import data_creation as dc
import constants as const
import utils


def pipeline():
    graph = dc.create_graph()
    model = dc.model_SIR_signal_propagation(graph)

    features = utils.one_hot_encode(list(model.status.values()), const.N_FEATURES)
    labels = utils.one_hot_encode(list(model.initial_status.values()), const.N_CLASSES)
    graph_structure = from_networkx(graph)

    model = GCN()
    train(model, graph_structure, features, labels)


if __name__ == "__main__":
    pipeline()

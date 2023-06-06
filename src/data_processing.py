from src import constants as const
import networkx as nx
import numpy as np
import os
import shutil
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import to_networkx
import torch


class SDDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [f"{i}.pt" for i in range(const.TRAINING_SIZE + const.VALIDATION_SIZE)]

    @property
    def processed_file_names(self):
        return [f"{i}.pt" for i in range(const.TRAINING_SIZE + const.VALIDATION_SIZE)]

    def process(self):
        for idx, raw_path in enumerate(self.raw_paths):
            # load raw data
            data = torch.load(raw_path)
            # process data
            if self.pre_filter is not None:
                data = self.pre_filter(data)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            # save data object
            torch.save(data, os.path.join(self.processed_dir, f"{idx}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"{idx}.pt"))
        return data


def paper_input(current_status: torch.tensor, edge_index: torch.tensor):
    """
    Prepares the input features for the GCNSI model according to the paper:
    https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    """
    Y = np.array(current_status)
    # S = get_laplacian(edge_index, normalization="sym")
    S = nx.normalized_laplacian_matrix(
        to_networkx(Data(edge_index=edge_index), to_undirected=True)
    )
    V3 = Y.copy()
    Y = [-1 if x == 0 else 1 for x in Y]
    V4 = [-1 if x == -1 else 0 for x in Y]
    I = np.identity(len(Y))
    a = const.ALPHA
    d1 = Y
    d2 = (1 - a) * np.linalg.inv(I - a * S).dot(Y)
    d3 = (1 - a) * np.linalg.inv(I - a * S).dot(V3)
    d4 = (1 - a) * np.linalg.inv(I - a * S).dot(V4)
    X = torch.from_numpy(np.column_stack((d1, d2, d3, d4))).float()
    return X


def create_distance_labels(
    graph: nx.Graph, initial_values: torch.tensor
) -> torch.tensor:
    """
    Creates the labels for the GCNR model. Each label is the distance of the node to the nearest source.
    """
    distances = []
    # extract all sources from prob_model
    sources = torch.where(initial_values == 1)[0].tolist()
    for source in sources:
        distances.append(nx.single_source_shortest_path_length(graph, source))
    # get min distance for each node
    min_distances = []
    for node in graph.nodes:
        min_distances.append(min([distance[node] for distance in distances]))

    return torch.tensor(np.expand_dims(min_distances, axis=1)).float()


def process_gcnsi_data(data: Data) -> Data:
    """Features and Labels for the GCNSI model."""
    X = paper_input(data.x, data.edge_index)
    y = torch.nn.functional.one_hot(data.y.to(torch.int64), const.N_CLASSES).float()
    return Data(x=X, y=y, edge_index=data.edge_index)


def process_gcnr_data(data: Data) -> Data:
    """Features and Labels for the GCNR model."""
    X = paper_input(data.x, data.edge_index)
    y = create_distance_labels(to_networkx(data, to_undirected=True), data.y)
    return Data(x=X, y=y, edge_index=data.edge_index)


def main():
    shutil.rmtree(os.path.join(const.DATA_PATH, "processed"), ignore_errors=True)


if __name__ == "__main__":
    main()
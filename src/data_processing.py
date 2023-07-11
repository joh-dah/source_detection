""" Creates new processed data based on the selected model. """
import argparse
import glob
from pathlib import Path
from src import constants as const
import networkx as nx
import numpy as np
import os
import shutil
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import to_networkx
import torch
import multiprocessing as mp
from tqdm import tqdm


def process_single(args):
    pre_transform, raw_path, idx, processed_dir = args
    # load raw data
    data = torch.load(raw_path)
    # process data
    if pre_transform is not None:
        data = pre_transform(data)
    # save data object
    torch.save(data, os.path.join(processed_dir, f"{idx}.pt"))


class SDDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.size = len(list((root / "raw").glob("*.pt")))
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [f"{i}.pt" for i in range(self.size)]

    @property
    def processed_file_names(self):
        return [f"{i}.pt" for i in range(self.size)]

    def process(self):
        # run in parallel using pool
        if self.pre_transform is not None:
            params = [
                (self.pre_transform, self.raw_paths[i], i, self.processed_dir)
                for i in range(self.size)
            ]
            with mp.Pool(const.N_CORES) as pool:
                print(f"Processing data set using multiprocessing ({pool})")
                list(
                    tqdm(
                        pool.imap_unordered(process_single, params),
                        total=self.size,
                    )
                )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"{idx}.pt"))
        return data


def paper_input(current_status: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
    """
    Prepares the input features for the GCNSI model according to the paper:
    https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    :param current_status: the current infection status
    :param edge_index: edge_index of a graph
    :return: prepared input features
    """
    Y = np.array(current_status)
    g = to_networkx(Data(edge_index=edge_index), to_undirected=False).to_undirected()
    S = nx.normalized_laplacian_matrix(g)
    V3 = Y.copy()
    Y = [-1 if x == 0 else 1 for x in Y]
    V4 = [-1 if x == -1 else 0 for x in Y]
    I = np.identity(len(Y))
    a = const.ALPHA
    d1 = Y
    temp = (1 - a) * np.linalg.inv(I - a * S)
    d2 = np.squeeze(np.asarray(temp.dot(Y)))
    d3 = np.squeeze(np.asarray(temp.dot(V3)))
    d4 = np.squeeze(np.asarray(temp.dot(V4)))
    X = torch.from_numpy(np.column_stack((d1, d2, d3, d4))).float()
    return X


def create_distance_labels(
    graph: nx.Graph, initial_values: torch.tensor
) -> torch.tensor:
    """
    Creates the labels for the GCNR model. Each label is the distance of the node to the nearest source.
    :param graph: graph for which to create the distance labels
    :param initial_values: initial values indicating the source nodes
    :return: distance labels
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
    """
    Features and Labels for the GCNSI model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = paper_input(data.x, data.edge_index)
    # expand labels to 2D tensor
    data.y = data.y.unsqueeze(1).float()
    return data


def process_simplified_gcnsi_data(data: Data) -> Data:
    """
    Simplified features and Labels for the GCNSI model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    # expand features to 2D tensor
    data.x = data.x.unsqueeze(1).float()
    # expand labels to 2D tensor
    data.y = data.y.unsqueeze(1).float()
    return data


def process_gcnr_data(data: Data) -> Data:
    """
    Features and Labels for the GCNR model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = paper_input(data.x, data.edge_index)
    data.y = create_distance_labels(to_networkx(data, to_undirected=True), data.y)
    return data


def main():
    """
    Creates new processed data based on the selected model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="whether to create validation or training data",
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic", help="name of the dataset"
    )
    args = parser.parse_args()

    train_or_val = "validation" if args.validation else "training"
    path = Path(const.DATA_PATH) / train_or_val / args.dataset.lower()

    print("Removing old processed data...")
    shutil.rmtree(path / "processed", ignore_errors=True)

    print("Creating new processed data...")
    if const.MODEL == "GCNSI":
        if const.SMALL_INPUT:
            pre_transform_function = process_simplified_gcnsi_data
        else:
            pre_transform_function = process_gcnsi_data
    elif const.MODEL == "GCNR":
        pre_transform_function = process_gcnr_data

    # triggers the process function of the dataset
    SDDataset(
        path,
        pre_transform=pre_transform_function,
    )


if __name__ == "__main__":
    main()

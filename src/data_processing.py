from src import constants as const
from src import utils
import networkx as nx
import numpy as np
import os
import pickle
from tqdm import tqdm
from pathlib import Path
from torch_geometric.data import Data
import torch


def paper_input(current_status: torch.tensor, nx_graph: nx.Graph):
    """
    Prepares the input features for the GCNSI model according to the paper:
    https://dl.acm.org/doi/abs/10.1145/3357384.3357994
    """
    Y = np.array(current_status)
    S = nx.normalized_laplacian_matrix(nx_graph)
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


def create_distance_labels(graph: nx.Graph, initial_values: torch.tensor):
    """
    Creates the labels for the GCNR model. Each label is the distance of the node to the nearest source.
    """
    distances = []
    # extract all sources from prob_model
    sources = np.where(np.array(initial_values) == 1)[0]
    for source in sources:
        distances.append(nx.single_source_shortest_path_length(graph, source))
    # get min distance for each node
    min_distances = []
    for node in graph.nodes:
        min_distances.append(min([distance[node] for distance in distances]))

    return torch.tensor(np.expand_dims(min_distances, axis=1)).float()


def process_gcnsi_data(
    current_status: torch.tensor, initial_status: torch.tensor, nx_graph: nx.Graph
):
    """Features and Labels for the GCNSI model."""
    X = paper_input(current_status, nx_graph)
    y = torch.nn.functional.one_hot(
        initial_status.to(torch.int64), const.N_CLASSES
    ).float()
    return X, y


def process_gcnr_data(
    current_status: torch.tensor, initial_status: torch.tensor, nx_graph: nx.Graph
):
    """Features and Labels for the GCNR model."""
    X = paper_input(current_status, nx_graph)
    y = create_distance_labels(nx_graph, initial_status)
    return X, y


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
    """
    for dataset in ["train", "validation"]:
        print(f"Processing {dataset} data")
        raw_path = f"{const.RAW_DATA_PATH}/{dataset}"
        processed_path = f"{const.PROCESSED_DATA_PATH}/{dataset}"
        Path(processed_path).mkdir(parents=True, exist_ok=True)
        for file_name in tqdm(os.listdir(raw_path)):
            raw_data = torch.load(f"{raw_path}/{file_name}")

            current_status = raw_data.x
            initial_status = raw_data.y
            edge_index = raw_data.edge_index

            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(range(len(current_status)))
            nx_graph.add_edges_from(edge_index.t().tolist())

            if const.MODEL == "GCNSI":
                X, y = process_gcnsi_data(current_status, initial_status, nx_graph)
            elif const.MODEL == "GCNR":
                X, y = process_gcnr_data(current_status, initial_status, nx_graph)

            processed_data = Data(x=X, y=y, edge_index=edge_index)
            torch.save(processed_data, f"{processed_path}/{file_name}")


if __name__ == "__main__":
    main()

""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import os
import random
import argparse
from typing import Optional
from pathlib import Path
import numpy as np
import ndlib.models.epidemics as epidemic_model
import ndlib.models.ModelConfig as mc
import networkx as nx
from tqdm import tqdm
import src.constants as const
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import to_networkx
import torch
import src.utils as utils


def select_random_sources(graph: nx.Graph, select_random: bool = True) -> list:
    """
    Selects nodes from the given graph as sources.
    The amount of nodes is randomly selected from a
    normal distribution with mean mean_sources and std mean_sources/2.
    :param graph: graph to select sources from
    :return: list of source nodes
    """
    if select_random:
        mu = const.MEAN_SOURCES
        sigma = int(np.sqrt(mu))
        n_sources = np.random.normal(mu, sigma)
        # make sure there are no sources smaller than 1 or larger than 1/4 of the graph
        n_sources = np.clip(n_sources, 1, len(graph.nodes) / 4).astype(int)
    else:
        n_sources = const.MEAN_SOURCES
    return random.choices(list(graph.nodes), k=n_sources)


def create_graph(graph_type: str) -> nx.Graph:
    """
    Creates a graph of the given type.
    :param graph_type: type of graph to create
    :return: created graph
    """
    n = np.random.normal(const.MEAN_N_NODES, np.sqrt(const.MEAN_N_NODES / 2))
    n = np.maximum(1, n).astype(int)

    if graph_type == "watts_strogatz":
        graph = nx.watts_strogatz_graph(n, const.WS_NEIGHBOURS, const.WS_PROBABILITY)
    elif graph_type == "barabasi_albert":
        graph = nx.barabasi_albert_graph(n, const.BA_NEIGHBOURS)
    else:
        raise ValueError("Unknown graph type")

    return graph


def create_signal_propagation_model(graph: nx.Graph, model_type: str) -> epidemic_model:
    """
    Creates a signal propagation model of the given type for the given graph.
    :param graph: graph to create the model for
    :param model_type: type of model to create
    :return: created model
    """
    source_nodes = select_random_sources(graph)
    beta = np.random.uniform(0, 1)

    config = mc.Configuration()
    if model_type == "SI":
        prop_model = epidemic_model.SIModel(graph)
        config.add_model_parameter("beta", beta)

    elif model_type == "SIR":
        gamma = np.random.uniform(0, beta)
        prop_model = epidemic_model.SIRModel(graph)
        config.add_model_parameter("beta", beta)
        config.add_model_parameter("gamma", gamma)

    else:
        raise ValueError("Unknown model type")

    config.add_model_initial_configuration("Infected", source_nodes)
    prop_model.set_initial_status(config)
    iterations = prop_model.iteration_bunch(const.ITERATIONS)
    prop_model.build_trends(iterations)

    return prop_model


def create_data_set(
    n_graphs: int,
    path: Path,
    existing_data: Optional[Dataset] = None,
    propagations_per_graph: int = 1,
    graph_type: str = const.GRAPH_TYPE,
    model_type: str = const.PROP_MODEL,
):
    """
    Creates n graphs of type graph_type and runs a
    signal propagation model of type model_type on them.
    The graphs and the results of the signal propagation are saved to the given path.
    :param n_graphs: number of graphs to create
    :param path: path to save the created data set to
    :param existing_data: existing data set, if supplied the signal propagation will be performed on the given graphs
    :param propagations_per_graph: number of signal propagations to perform per graph
    :param graph_type: type of graph to create
    :param model_type: type of model to use for signal propagation
    """

    path /= "raw"
    Path(path).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))

    for i in tqdm(range(n_graphs), disable=const.ON_CLUSTER):
        if existing_data is None:
            graph = create_graph(graph_type)
        else:
            graph = to_networkx(existing_data[i], to_undirected=False).to_undirected()
        edge_index = (
            torch.tensor(list(graph.to_directed().edges), dtype=torch.long)
            .t()
            .contiguous()
        )
        for j in range(propagations_per_graph):
            prop_model = create_signal_propagation_model(graph, model_type)
            X = torch.tensor(list(prop_model.status.values()), dtype=torch.float)
            y = torch.tensor(
                list(prop_model.initial_status.values()), dtype=torch.float
            )
            data = Data(x=X, y=y, edge_index=edge_index)
            data.validate()
            torch.save(data, path / f"{i * propagations_per_graph + j}.pt")


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
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

    if args.validation:
        if args.dataset == "synthetic":
            print("Create Synthetic Validation Data:")
            create_data_set(const.VALIDATION_SIZE, path)
        else:
            print(f"{args.dataset} Validation Data:")
            dataset = utils.get_dataset_from_name(args.dataset)
            create_data_set(
                len(dataset),
                path,
                dataset,
                propagations_per_graph=const.PROPAGATIONS_PER_REAL_WORLD_GRAPH,
            )
    else:
        print("Create Synthetic Train Data:")
        create_data_set(const.TRAINING_SIZE, path)


if __name__ == "__main__":
    main()

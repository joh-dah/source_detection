""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import os
import random
import pickle
from pathlib import Path
import numpy as np
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
from tqdm import tqdm
import src.constants as const
from ndlib.models.DiffusionModel import DiffusionModel
from torch_geometric.data import Data
import torch


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


def create_signal_propagation_model(graph: nx.Graph, model_type: str):
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
        prop_model = ep.SIModel(graph)
        config.add_model_parameter("beta", beta)

    elif model_type == "SIR":
        gamma = np.random.uniform(0, beta)
        prop_model = ep.SIRModel(graph)
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
    graph_type: str = const.GRAPH_TYPE,
    model_type: str = const.PROP_MODEL,
):
    """
    Creates n graphs of type graph_type and runs a
    signal propagation model of type model_type on them.
    The graphs and the results of the signal propagation are saved to the given path.
    :param n_graphs: number of graphs to create
    :param graph_type: type of graph to create
    :param model_type: type of model to use for signal propagation
    """

    path = Path(const.RAW_DATA_PATH)
    Path(path).mkdir(parents=True, exist_ok=True)
    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))

    for i in tqdm(range(n_graphs), disable=const.ON_CLUSTER):
        graph = create_graph(graph_type)
        prop_model = create_signal_propagation_model(graph, model_type)
        X = torch.tensor(list(prop_model.status.values()), dtype=torch.float)
        y = torch.tensor(list(prop_model.initial_status.values()), dtype=torch.float)
        edge_index = (
            torch.tensor(list(graph.to_directed().edges), dtype=torch.long)
            .t()
            .contiguous()
        )
        data = Data(x=X, y=y, edge_index=edge_index)
        data.validate()
        torch.save(data, path / f"{i}.pt")


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
    """
    print("Create Data:")
    create_data_set(const.TRAINING_SIZE + const.VALIDATION_SIZE)


if __name__ == "__main__":
    main()

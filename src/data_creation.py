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


def select_random_sources(graph):
    """
    Selects nodes from the given graph as sources.
    The amount of nodes is randoly selected from a
    normal distribution with mean mean_sources and std mean_sources/2.
    :param graph: graph to select sources from
    :return: list of source nodes
    """
    mu = const.MEAN_SOURCES
    sigma = int(np.sqrt(mu))
    n_sources = np.random.normal(mu, sigma)
    # make sure there are no sources smaller than 1 or larger than 1/4 of the graph
    n_sources = np.clip(n_sources, 1, len(graph.nodes) / 4).astype(int)
    return random.choices(list(graph.nodes), k=n_sources)


def model_SIR_signal_propagation(graph, iterations, seed=None):
    """
    Creates a SIR model and runs it on the given graph.
    :param graph: graph to run the model on
    :param seed: seed for signal propagation
    :param iterations: number of iterations to run the model for
    :return: propagation model
    """
    model = ep.SIRModel(graph, seed=seed)
    source_nodes = select_random_sources(graph)

    config = mc.Configuration()
    config.add_model_parameter("beta", const.SIR_BETA)
    config.add_model_parameter("gamma", const.SIR_GAMMA)
    config.add_model_initial_configuration("Infected", source_nodes)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(iterations)
    model.build_trends(iterations)

    return model


def model_SI_signal_propagation(graph, iterations, seed=None):
    """
    Creates a SI model and runs it on the given graph.
    :param graph: graph to run the model on
    :param seed: seed for signal propagation
    :param iterations: number of iterations to run the model for
    :return: propagation model
    """
    model = ep.SIModel(graph, seed=seed)
    source_nodes = select_random_sources(graph)

    config = mc.Configuration()
    config.add_model_parameter("beta", const.SI_BETA)
    config.add_model_initial_configuration("Infected", source_nodes)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(iterations)
    model.build_trends(iterations)

    return model


def create_graph(graph_type):
    """
    Creates a graph of the given type.
    :param graph_type: type of graph to create
    :param n_nodes: number of nodes in the graph
    :return: created graph
    """
    n = np.random.normal(const.MEAN_N_NODES, np.sqrt(const.MEAN_N_NODES))
    n = np.maximum(1, n).astype(int)

    if graph_type == "watts_strogatz":
        graph = nx.watts_strogatz_graph(n, const.WS_NEIGHBOURS, const.WS_PROBABILITY)
    elif graph_type == "barabasi_albert":
        graph = nx.barabasi_albert_graph(n, const.BA_NEIGHBOURS)
    else:
        raise ValueError("Unknown graph type")

    return graph


def create_signal_propagation_model(graph, model_type):
    """
    Creates a signal propagation model of the given type for the given graph.
    :param graph: graph to create the model for
    :param model_type: type of model to create
    :param iterations: number of iterations to run the model for
    :return: created model
    """
    iterations = np.random.normal(const.MEAN_ITERS, int(np.sqrt(const.MEAN_ITERS)))
    iterations = np.maximum(1, iterations).astype(int)

    if model_type == "SI":
        prop_model = model_SI_signal_propagation(graph, iterations)
    elif model_type == "SIR":
        prop_model = model_SIR_signal_propagation(graph, iterations)
    else:
        raise ValueError("Unknown model type")

    return prop_model


def create_data_set(
    n_graphs, path, graph_type=const.GRAPH_TYPE, model_type=const.PROP_MODEL
):
    """
    Creates n graphs of type graph_type and runs a
    signal propagation model of type model_type on them.
    The graphs and the results of the signal propagation are saved to the given path.
    :param n: number of graphs to create
    :param path: path to save the data set to
    :param graph_type: type of graph to create
    :param model_type: type of model to use for signal propagation
    """

    path = Path(f"{const.DATA_PATH}/{path}")
    Path(path).mkdir(parents=True, exist_ok=True)
    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))

    for i in tqdm(range(n_graphs)):
        graph = create_graph(graph_type)
        prop_model = create_signal_propagation_model(graph, model_type)
        pickle.dump(prop_model, open(path / f"{i}.pkl", "wb"))


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
    """
    print("Create Train Data:")
    create_data_set(const.TRAINING_SIZE, "train")
    print("Create Validation Data:")
    create_data_set(const.VALIDATION_SIZE, "validation")


if __name__ == "__main__":
    main()

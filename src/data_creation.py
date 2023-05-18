import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import src.utils as utils
import src.constants as const
import random
import pickle
from pathlib import Path
from tqdm import tqdm


def select_random_sources(graph, n):
    """
    Selects n random nodes from the given graph as sources.
    :param graph: graph to select sources from
    :param n: number of sources to select
    :return: list of source nodes
    """
    return random.choices(list(graph.nodes), k=n)


def model_SIR_signal_propagation(graph, seed=None):
    """
    Creates a SIR model and runs it on the given graph.
    :param graph: graph to run the model on
    :param seed: seed for signal propagation
    :return: propagation model
    """
    model = ep.SIRModel(graph, seed=seed)
    source_nodes = select_random_sources(graph, const.N_SOURCES)

    config = mc.Configuration()
    config.add_model_parameter("beta", const.SIR_BETA)
    config.add_model_parameter("gamma", const.SIR_GAMMA)
    config.add_model_initial_configuration("Infected", source_nodes)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(const.N_ITERATIONS)
    model.build_trends(iterations)

    return model


def model_SI_signal_propagation(graph, seed=None):
    """
    Creates a SI model and runs it on the given graph.
    :param graph: graph to run the model on
    :param seed: seed for signal propagation
    """
    model = ep.SIModel(graph, seed=seed)
    source_nodes = select_random_sources(graph, const.N_SOURCES)

    config = mc.Configuration()
    config.add_model_parameter("beta", const.SI_BETA)
    config.add_model_initial_configuration("Infected", source_nodes)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(const.N_ITERATIONS)
    model.build_trends(iterations)

    return model


def create_data_set(n, path, graph_type="watts_strogatz", model_type="SIR"):
    """
    Creates n graphs of type graph_type and runs a signal propagation model of type model_type on them.
    The graphs and the results of the signal propagation are saved to the given path.
    :param n: number of graphs to create
    :param path: path to save the data set to
    :param graph_type: type of graph to create
    :param model_type: type of model to use for signal propagation
    """
    path = Path(f"{const.DATA_PATH}/{path}")
    Path(path).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n)):
        if graph_type == "watts_strogatz":
            graph = nx.watts_strogatz_graph(
                const.N_NODES, const.WS_NEIGHBOURS, const.WS_PROBABILITY
            )
        elif graph_type == "barabasi_albert":
            graph = nx.barabasi_albert_graph(const.N_NODES, const.BA_NEIGHBOURS)
        else:
            raise AssertionError("Unknown graph type")

        if model_type == "SI":
            prop_model = model_SI_signal_propagation(graph)
        elif model_type == "SIR":
            prop_model = model_SIR_signal_propagation(graph)
        else:
            raise AssertionError("Unknown model type")

        pickle.dump(prop_model, open(path / f"{i}.pkl", "wb"))


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
    """
    print("Create Train Data:")
    create_data_set(40, "train", const.GRAPH_TYPE, const.PROP_MODEL)
    print("Create Validation Data:")
    create_data_set(40, "validation", const.GRAPH_TYPE, const.PROP_MODEL)


if __name__ == "__main__":
    main()

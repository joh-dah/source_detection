import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from torch_geometric.utils.convert import from_networkx
import utils
import constants as const
import random


def create_graph(seed=None):
    return nx.watts_strogatz_graph(const.N_NODES, 8, 0.2, seed=seed)


def select_random_sources(graph, n_sources):
    return random.sample(graph.nodes, n_sources)


def model_SIR_signal_propagation(graph, seed=None):
    model = ep.SIRModel(graph, seed=seed)
    source_nodes = select_random_sources(graph, const.N_SOURCES)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("beta", const.SIR_BETA)
    config.add_model_parameter("gamma", const.SIR_GAMMA)
    config.add_model_initial_configuration("Infected", source_nodes)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(const.N_ITERATIONS)
    model.build_trends(iterations)

    return model


def create_data_set(n_graphs):
    data = []
    for i in range(n_graphs):
        g = create_graph()
        m = model_SIR_signal_propagation(g)
        graph_structure = from_networkx(g)
        features = utils.one_hot_encode(list(m.status.values()), const.N_FEATURES)
        labels = utils.one_hot_encode(list(m.initial_status.values()), const.N_CLASSES)
        data.append([graph_structure, features, labels])
    return data

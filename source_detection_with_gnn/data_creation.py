import networkx as nx
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc


def create_graph():
    return nx.erdos_renyi_graph(100, 0.1, seed=123)


def model_SIR_signal_propagation(graph):
    model = ep.SIRModel(graph)

    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("beta", 0.006)
    config.add_model_parameter("gamma", 0.01)
    config.add_model_parameter("fraction_infected", 0.1)

    model.set_initial_status(config)
    iterations = model.iteration_bunch(30)
    model.build_trends(iterations)

    return model

import numpy as np
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
import time
import pandas as pd

RELATIVE_N_SOURCES_RANGE = (0.01, 0.05)  # (0.02, 0.0200001)
RELATIVE_INFECTED_RANGE = (0.20, 0.80)  # (0.8, 0.800001)
N_NODES_RANGE = (100, 10000)  # (10000, 10001)
WS_NEIGHBOURS_RANGE = (0.001, 0.05)  # (0.01, 0.0100001)
WS_PROBABILITY_RANGE = (0.0, 0.5)  # (0.5, 0.500001)

BETA_RANGE = (0.01, 0.4)  # (0.01, 0.0100001)

root_seed = 0x8C3C010CB4754C905776BDA35EE7501

# RELATIVE_N_SOURCES_RANGE = (0.02, 0.0200001)  # (0.01, 0.05)
# RELATIVE_INFECTED_RANGE = (0.8, 0.800001)  # (0.20, 0.80)
# N_NODES_RANGE = (10000, 10001)  # (100, 10000)
# WS_NEIGHBOURS_RANGE = (0.01, 0.0100001)  # (0.001, 0.01)
# WS_PROBABILITY_RANGE = (0.5, 0.500001)  # (0.0, 0.5)

# BETA_RANGE = (0.01, 0.0100001)  # (0.01, 0.9)


def random_generator(seed):
    return np.random.default_rng([seed, root_seed])


def create_graph(seed: int) -> nx.Graph:
    """
    Creates a graph of the given type.
    :return: created graph
    """
    n = random_generator(seed).integers(*N_NODES_RANGE)
    neighbours = random_generator(seed + 1).uniform(*WS_NEIGHBOURS_RANGE) * n
    neighbours = np.clip(neighbours, 2, n).astype(int)
    prob_reconnect = random_generator(seed + 2).uniform(*WS_PROBABILITY_RANGE)
    graph_type = random_generator(seed + 3).choice(
        ["watts_strogatz", "barabasi_albert"]
    )
    if graph_type == "watts_strogatz":
        graph = nx.watts_strogatz_graph(n, neighbours, prob_reconnect)
    elif graph_type == "barabasi_albert":
        neighbours = int(neighbours / 2)
        graph = nx.barabasi_albert_graph(n, neighbours)

    return graph, graph_type, neighbours, prob_reconnect


def iterate_until(threshold_infected: float, model: ep.SIModel) -> int:
    """
    Runs the given model until the given percentage of nodes is infected.
    :param threshold_infected: maximum percentage of infected nodes
    :param model: model to run
    :param config: configuration of the model
    :return: number of iterations until threshold was reached
    """

    iterations = 0
    threshold = int(threshold_infected * len(model.status))
    n_infected_nodes = sum([x if x == 1 else 0 for x in model.status.values()])
    while n_infected_nodes <= threshold and iterations < 10:
        n_infected_nodes = sum([x if x == 1 else 0 for x in model.status.values()])
        model.iteration(False)
        iterations += 1

    return iterations


def signal_propagation(seed: int, graph: nx.Graph):
    """
    Simulates signal propagation on the given graph.
    :param graph: graph to simulate signal propagation on
    :return: list of infected nodes
    """
    model = ep.SIModel(graph)
    config = mc.Configuration()
    beta = random_generator(seed).uniform(*BETA_RANGE)
    config.add_model_parameter("beta", beta)
    percentage_infected = random_generator(seed + 1).uniform(*RELATIVE_N_SOURCES_RANGE)
    config.add_model_parameter("percentage_infected", percentage_infected)
    model.set_initial_status(config)
    threshold_infected = random_generator(seed + 2).uniform(*RELATIVE_INFECTED_RANGE)
    iterations = iterate_until(threshold_infected, model)
    return model, iterations, percentage_infected, beta, threshold_infected


def experiment(i):
    """
    Runs the experiment.
    """
    seed = i * 10
    graph, graph_type, neighbours, prob_reconnect = create_graph(seed)
    (
        model,
        iterations,
        percentage_infected,
        beta,
        threshold_infected,
    ) = signal_propagation(seed + 5, graph)
    n_sources = sum([x if x == 1 else 0 for x in model.initial_status.values()])
    n_infected_nodes = sum([x if x == 1 else 0 for x in model.status.values()])
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    avg_degree = n_edges / n_nodes
    avg_clustering = nx.average_clustering(graph)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "avg_degree": avg_degree,
        "avg_clustering": avg_clustering,
        "n_sources": n_sources,
        "n_infected_nodes": n_infected_nodes,
        "n_iterations": iterations,
        "percentage_infected": percentage_infected,
        "beta": beta,
        "threshold_infected": threshold_infected,
        "graph_type": graph_type,
        "neighbours": neighbours,
        "prob_reconnect": prob_reconnect,
    }


def run_experiments(n_experiments: int):
    """
    Runs the experiment n_experiments times.
    :param n_experiments: number of experiments to run
    :return: list of results
    """
    with mp.Pool(999) as pool:
        print(pool)
        results = list(
            tqdm(
                pool.imap_unordered(experiment, range(n_experiments)),
                total=n_experiments,
            )
        )

    pd.DataFrame(results).to_csv("results1.csv", index=False)
    return results


def run_experiments_single_process(n_experiments: int):
    """
    Runs the experiment n_experiments times.
    :param n_experiments: number of experiments to run
    :return: list of results
    """
    results = []
    for i in range(n_experiments):
        results.append(experiment(i))
    return results


if __name__ == "__main__":
    # track the time it takes to run the experiment
    start = time.time()
    run_experiments(10000)
    end = time.time()
    print(f"Time elapsed: {end - start}")

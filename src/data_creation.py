""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import os
import random
import argparse
from typing import Optional
from pathlib import Path
import numpy as np
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
from tqdm import tqdm
import src.constants as const
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import to_networkx
import torch
import src.utils as utils
import multiprocessing as mp
import torch_geometric.transforms as T


def random_generator(seed):
    return np.random.default_rng([seed, const.ROOT_SEED])


def create_graph(seed: int) -> nx.Graph:
    """
    Creates a graph of the given type.
    :return: created graph
    """
    n = random_generator(seed).integers(*const.N_NODES)
    neighbours = (
        random_generator(seed + 1).uniform(*const.WATTS_STROGATZ_NEIGHBOURS) * n
    )
    neighbours = np.clip(neighbours, 2, n).astype(int)
    prob_reconnect = random_generator(seed + 2).uniform(
        *const.WATTS_STROGATZ_PROBABILITY
    )
    graph_type = random_generator(seed + 3).choice(
        ["watts_strogatz", "barabasi_albert"]
    )
    success = False
    iterations = 0
    while not success:
        if graph_type == "watts_strogatz":
            graph = nx.watts_strogatz_graph(
                n, neighbours, prob_reconnect, seed=seed + 4 + iterations
            )
        elif graph_type == "barabasi_albert":
            neighbours = int(neighbours / 2)
            graph = nx.barabasi_albert_graph(n, neighbours, seed=seed + 4 + iterations)
        success = nx.is_connected(graph)
        iterations += 1
        prob_reconnect /= 2
        if iterations > 10:
            raise Exception("Could not create connected graph.")

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
    while n_infected_nodes <= threshold and iterations < 50:
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
    model = ep.SIModel(graph, seed=seed + 1)
    config = mc.Configuration()
    beta = random_generator(seed).uniform(*const.BETA)
    config.add_model_parameter("beta", beta)
    percentage_infected = random_generator(seed + 2).uniform(*const.RELATIVE_N_SOURCES)
    config.add_model_parameter("percentage_infected", percentage_infected)
    model.set_initial_status(config)
    threshold_infected = random_generator(seed + 3).uniform(*const.RELATIVE_INFECTED)
    iterations = iterate_until(threshold_infected, model)
    return model, iterations, percentage_infected, beta, threshold_infected


def generate_metrics(graph: nx.Graph, data: Data):
    """
    Generates metrics for the given graph and model.
    :param graph: graph to generate metrics for
    :param model: model to generate metrics for
    :param data: data point to save metrics to
    """
    data.metrics = dict(
        # diameter=nx.diameter(graph),
        # average_shortest_path_length=nx.average_shortest_path_length(
        #     graph, method="unweighted"
        # ),
        average_clustering_coefficient=nx.average_clustering(graph),
        average_degree=np.mean([x[1] for x in graph.degree]),
        n_nodes=len(graph.nodes),
        n_edges=len(graph.edges),
        avg_degree_centrality=np.mean(list(nx.degree_centrality(graph).values())),
    )


def create_data(params: tuple):
    """
    Creates a single data point. Consisting of a graph and the result of a signal propagation model on that graph.
    The data point is saved to the given path.
    :param i: index of the data point (used for seeding)
    :param file: path to save the data point to (including file name)
    :param existing_data: existing data point, if supplied the signal propagation will be performed on the given graph
    """

    i, path, existing_data, metrics = params

    seed = i * 20
    if existing_data is not None:
        graph = to_networkx(
            existing_data, to_undirected=False, remove_self_loops=True
        )
        transform = T.Compose([T.LargestConnectedComponents, T.ToUndirected])
        graph = transform(graph)
        graph_type = "real world"
        neighbours = -1
        prob_reconnect = -1

    else:
        graph, graph_type, neighbours, prob_reconnect = create_graph(seed)
    (
        prop_model,
        iterations,
        percentage_infected,
        beta,
        threshold_infected,
    ) = signal_propagation(seed + 15, graph)
    X = torch.tensor(list(prop_model.status.values()), dtype=torch.float)
    y = torch.tensor(list(prop_model.initial_status.values()), dtype=torch.float)
    edge_index = (
        torch.tensor(list(graph.to_directed().edges), dtype=torch.long).t().contiguous()
    )
    data = Data(
        x=X,
        y=y,
        edge_index=edge_index,
        settings=dict(
            graph_type=graph_type,
            neighbours=neighbours,
            prob_reconnect=prob_reconnect,
            beta=beta,
            threshold_infected=threshold_infected,
            iterations=iterations,
            percentage_initially_infected=percentage_infected,
            currently_infected=sum(
                [x if x == 1 else 0 for x in prop_model.status.values()]
            ),
        ),
    )
    if metrics:
        generate_metrics(graph, data)
    torch.save(data, path / f"{i}.pt")


def create_data_set(
    n_graphs: int,
    path: Path,
    existing_data: Optional[Dataset] = None,
    propagations_per_graph: int = 1,
    generate_graph_metrics: bool = True,
):
    """
    Creates n random graphs and performs a signal propagation on each of them.
    Parameters of the signal propagation are chosen randomly from ranges given in params.yaml.
    The graphs and the results of the signal propagation are saved to the given path.
    :param n_graphs: number of graphs to create
    :param path: path to save the created data set to
    :param existing_data: existing dataset, if supplied the signal propagation will be performed on the given graphs
    :param propagations_per_graph: number of signal propagations to perform per graph
    """

    path /= "raw"
    Path(path).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))

    inputs = [
        (
            i * propagations_per_graph + j,
            path,
            existing_data[i] if existing_data is not None else None,
            j == 0 and generate_graph_metrics,
        )
        for j in range(propagations_per_graph)
        for i in range(n_graphs)
    ]

    with mp.Pool(const.N_CORES) as pool:
        print(f"Creating data set using multiprocessing ({pool})")
        list(
            tqdm(
                pool.imap_unordered(create_data, inputs),
                total=len(inputs),
            )
        )


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

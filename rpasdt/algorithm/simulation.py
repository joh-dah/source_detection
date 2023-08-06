"""Experiment simulation utilities."""
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List

from torch_geometric.utils.convert import to_networkx

import networkx as nx
from ndlib.models import DiffusionModel
from tqdm import tqdm
import multiprocessing as mp

from rpasdt.algorithm.diffusion import (
    get_and_init_diffusion_model,
    get_nodes_by_diffusion_status,
)
from rpasdt.algorithm.graph_loader import load_graph
from rpasdt.algorithm.models import (
    DiffusionSimulationConfig,
    DiffusionSimulationResult,
    SourceDetectionSimulationConfig,
    SourceDetectionSimulationResult,
)
from rpasdt.algorithm.source_detectors.source_detection import (
    get_source_detector,
)
from rpasdt.algorithm.source_selection import select_sources
from rpasdt.algorithm.taxonomies import NodeStatusEnum


@dataclass
class DiffusionSimulation:
    experiment_number: int
    graph: nx.Graph
    source_nodes: List[int]
    diffusion_model: DiffusionModel
    diffusion_iterations: List[Dict]

    @cached_property
    def diffusion_trends(self):
        return self.diffusion_model.build_trends(self.diffusion_iterations)


def _simulate_diffusion(simulation_config: DiffusionSimulationConfig):
    for experiment_number in range(simulation_config.number_of_experiments):
        graph: nx.Graph = simulation_config.graph or load_graph(
            graph_type=simulation_config.graph_type,
            graph_type_properties=simulation_config.graph_type_properties,
        )
        source_nodes: List[int] = simulation_config.source_nodes or select_sources(
            config=simulation_config.source_selection_config, graph=graph
        )

        diffusion_models: List[DiffusionModel] = [
            get_and_init_diffusion_model(
                graph=graph,
                source_nodes=source_nodes,
                diffusion_type=diffusion_model.diffusion_model_type,
                model_params=diffusion_model.diffusion_model_params,
            )[0]
            for diffusion_model in simulation_config.diffusion_models
        ]
        for diffusion_model in diffusion_models:
            diffusion_iterations = diffusion_model.iteration_bunch(
                simulation_config.iteration_bunch
            )
            yield DiffusionSimulation(
                experiment_number=experiment_number,
                graph=graph,
                source_nodes=source_nodes,
                diffusion_model=diffusion_model,
                diffusion_iterations=diffusion_iterations,
            )


def perform_diffusion_simulation(
    simulation_config: DiffusionSimulationConfig,
) -> DiffusionSimulationResult:
    result = DiffusionSimulationResult(simulation_config=simulation_config)

    for simulation in _simulate_diffusion(simulation_config):
        result.add_result(
            diffusion_model=simulation.diffusion_model,
            iterations=simulation.diffusion_iterations,
            trends=simulation.diffusion_trends,
        )
    return result


def _perform_source_detection(args):
    data, source_detection_config = args
    G = to_networkx(data, to_undirected=True)
    infected_nodes = data.x.nonzero().flatten().tolist()
    IG = G.subgraph(infected_nodes)
    source_nodes = data.y.nonzero().flatten().tolist()
    for (
        name,
        source_detector_config,
    ) in source_detection_config.source_detectors.items():
        source_detector = get_source_detector(
            algorithm=source_detector_config.alg,
            G=G,
            IG=IG,
            config=source_detector_config.config,
            number_of_sources=len(source_nodes),
        )
        sd_evaluation = source_detector.evaluate_sources(source_nodes)
        return name, source_detector_config, sd_evaluation


def perform_source_detection_simulation(
    source_detection_config: SourceDetectionSimulationConfig, data_set
) -> SourceDetectionSimulationResult:
    result = SourceDetectionSimulationResult(
        source_detection_config=source_detection_config
    )
    with mp.Pool() as pool:
        print(pool)
        args = [(data, source_detection_config) for data in data_set]
        results = list(
            tqdm(
                pool.imap_unordered(_perform_source_detection, args),
                total=len(data_set),
            )
        )
        for name, source_detector_config, sd_evaluation in results:
            result.add_result(name, source_detector_config, sd_evaluation)
    return result


# result = perform_diffusion_simulation(DiffusionSimulationConfig(
#     diffusion_models=[DiffusionModelSimulationConfig(
#         diffusion_model_type=DiffusionTypeEnum.SI,
#         diffusion_model_params={"beta": 0.08784399402913001}
#     )]
# ))
# print(result.avg_iteration_to_status_in_population(NodeStatusEnum.INFECTED))

# sample source detection
# result = perform_source_detection_simulation(SourceDetectionSimulationConfig(
#
#     diffusion_models=[DiffusionModelSimulationConfig(
#         diffusion_model_type=DiffusionTypeEnum.SI,
#         diffusion_model_params={"beta": 0.08784399402913001}
#     )],
#     iteration_bunch=20,
#     source_selection_config=NetworkSourceSelectionConfig(
#         number_of_sources=5,
#     ),
#     source_detectors={
#         "NETLSEUTH": SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.NET_SLEUTH,
#             config=CommunitiesBasedSourceDetectionConfig()
#         ),
#         "RUMOR_CENTER": SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.RUMOR_CENTER,
#             config=CommunitiesBasedSourceDetectionConfig()
#         )
#     }
# ))
# print(result.aggregated_results)
# for name, results in result.raw_results.items():
#     for mm_r in results:
#         print(
#             f'{name}-{mm_r.real_sources}-{mm_r.detected_sources}-{mm_r.TP}:{mm_r.FN}:{mm_r.FP}')

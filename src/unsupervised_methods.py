import rpasdt.algorithm.models as rpasdt_models
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import DiffusionTypeEnum, SourceDetectionAlgorithm
from src.data_processing import SDDataset, process_gcnr_data
from src.validation import min_matching_distance
import src.constants as const
import torch
import numpy as np


def create_simulation_config():
    return rpasdt_models.SourceDetectionSimulationConfig(
        diffusion_models=[
            rpasdt_models.DiffusionModelSimulationConfig(
                diffusion_model_type=DiffusionTypeEnum.SI,
                diffusion_model_params={"beta": 0.08784399402913001},
            )
        ],
        iteration_bunch=20,
        source_selection_config=rpasdt_models.NetworkSourceSelectionConfig(
            number_of_sources=5,
        ),
        source_detectors={
            "NETLSEUTH": rpasdt_models.SourceDetectorSimulationConfig(
                alg=SourceDetectionAlgorithm.NET_SLEUTH,
                config=rpasdt_models.CommunitiesBasedSourceDetectionConfig(),
            )
        },
    )


def main():
    val_data = SDDataset(const.DATA_PATH, pre_transform=process_gcnr_data)
    raw_data_paths = val_data.raw_paths[const.TRAINING_SIZE :]

    val_data = []
    for path in raw_data_paths:
        val_data.append(torch.load(path))

    simulation_config = create_simulation_config()

    result = perform_source_detection_simulation(simulation_config, val_data)
    print(result)
    print(result.aggregated_results)
    for name, results in result.raw_results.items():
        for mm_r in results:
            print(f"{name}-{mm_r.real_sources}-{mm_r.detected_sources}-{mm_r.TP}:{mm_r.FN}:{mm_r.FP}")


if __name__ == "__main__":
    main()

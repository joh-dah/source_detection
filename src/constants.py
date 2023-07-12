"""Global constants for the project."""
import yaml

params = yaml.full_load(open("params.yaml", "r"))

# General
MODEL = params["model"]  # "GCNR" or "GCNSI"
MODEL_NAME = params["model_name"]  # defins
DATA_PATH = "data"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH = "models"
FIGURES_PATH = "figures"
ROC_PATH = "roc"
REPORT_PATH = "reports"
ON_CLUSTER = params["on_cluster"]
N_CORES = 999

# Data Creation
dc = params["data_creation"]
TRAINING_SIZE = dc["training_size"]
VALIDATION_SIZE = dc["validation_size"]
GRAPH_TYPE = dc["graph_type"]
SMALL_INPUT = dc["small_input"]  # "true" or "false"
PROPAGATIONS_PER_REAL_WORLD_GRAPH = dc["propagations_per_real_world_graph"]
RELATIVE_N_SOURCES = tuple(dc["relative_n_sources"])
RELATIVE_INFECTED = tuple(dc["relative_infected"])
N_NODES = tuple(dc["n_nodes"])
WATTS_STROGATZ_NEIGHBOURS = tuple(dc["watts_strogatz_neighbours"])
WATTS_STROGATZ_PROBABILITY = tuple(dc["watts_strogatz_probability"])
BETA = tuple(dc["beta"])
ROOT_SEED = dc["root_seed"]

# Training
training = params["training"]
EPOCHS = training["epochs"]
LEARNING_RATE = training["learning_rate"]
HIDDEN_SIZE = training["hidden_size"]
LAYERS = training["layers"]
ALPHA = training["alpha"]
WEIGHT_DECAY = training["weight_decay"]
BATCH_SIZE = training["batch_size"]
USE_LOG_LOSS = training["useLogLoss"]
SUBSAMPLE = training["subsample"]
GCNR_N_FEATURES = 4
if SMALL_INPUT:
    GCNSI_N_FEATURES = 1
else:
    GCNSI_N_FEATURES = 4

# Visualization
SEED = params["visualization"]["seed"]

"""Global constants for the project."""
import json

params = json.load(open("params.json", "r"))

# General
MODEL = params["model"]  # "GCNR" or "GCNSI"
DATA_PATH = "data"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_PATH = "models"
FIGURES_PATH = "figures"
ROC_PATH = "roc"
REPORT_PATH = "reports"

# Data Creation
dc = params["data_creation"]
GRAPH_TYPE = dc["graph_type"]  # "watts_strogatz" or "barabasi_albert"
PROP_MODEL = dc["prop_model"]  # "SIR" or "SI"
TRAINING_SIZE = dc["training_size"]
VALIDATION_SIZE = dc["validation_size"]
MEAN_N_NODES = dc["mean_nodes"]
WS_NEIGHBOURS = dc["ws_neighbours"]
WS_PROBABILITY = dc["ws_probability"]
BA_NEIGHBOURS = dc["ba_neighbours"]
MEAN_SOURCES = dc["mean_sources"]
MEAN_ITERS = dc["mean_iters"]
SIR_BETA = dc["sir_beta"]
SIR_GAMMA = dc["sir_gamma"]
SI_BETA = dc["si_beta"]
SMALL_INPUT = dc["small_input"]

# Training
training = params["training"]
EPOCHS = training["epochs"]
LEARNING_RATE = training["learning_rate"]
HIDDEN_SIZE = training["hidden_size"]
LAYERS = training["layers"]
ALPHA = training["alpha"]
WEIGHT_DECAY = training["weight_decay"]
GCNR_N_FEATURES = 4
if SMALL_INPUT:
    GCNSI_N_FEATURES = 1
else:
    GCNSI_N_FEATURES = 4

# Visualization
SEED = params["visualization"]["seed"]

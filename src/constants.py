"""Global constants for the project."""
# Run Parameters
PROP_MODEL = "SIR"  # "SIR" or "SI"
GRAPH_TYPE = "barabasi_albert"  # "watts_strogatz" or "barabasi_albert"
MODEL = "GCNR"  # "GCN" or "GCNSI"

# General
MODEL_PATH = "models"
FIGURES_PATH = "figures"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
SEED = 123

# Model Constants
TRAINING_SIZE = 100
VALIDATION_SIZE = 20
EPOCHS = 20
LEARNING_RATE = 0.000001
HIDDEN_SIZE = 512
LAYERS = 5
N_FEATURES = 3
N_CLASSES = 2  # is source/ not source
## GCNSI
ALPHA = 0.5  # controls the influence that a node gets from its neighbors in feature creation
GCNSI_N_FEATURES = 4
GCNR_N_FEATURES = 4
WEIGHT_DECAY = 0.1


# Graph Constants
MEAN_N_NODES = 30  # mean of number of nodes in the graph
## Watts-Strogatz
WS_NEIGHBOURS = 10  # control number of neighbours to connect to
WS_PROBABILITY = 0.2  # probability of rewiring an edge
## Barabasi-Albert
BA_NEIGHBOURS = 2  # control number of edges to attach between new and existing nodes


# Signal Propagation Constants
MEAN_SOURCES = 2  # number of infection sources (probably make random later)
MEAN_ITERS = 65  # number of iterations for signal propagation
## SIR
SIR_BETA = 0.01  # probability for neighbor to get infected
SIR_GAMMA = 0.005  # probability of node to recover
## SI
SI_BETA = 0.01  # probability for neighbor to get infected

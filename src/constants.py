"""Global constants for the project."""
# Run Parameters
PROP_MODEL = "SIR"  # "SIR" or "SI"
GRAPH_TYPE = "watts_strogatz"  # "watts_strogatz" or "barabasi_albert"
MODEL = "GCNSI"  # "GCN" or "GCNSI"

# General
MODEL_PATH = "models"
FIGURES_PATH = "figures"
DATA_PATH = "data"
SEED = 123

# Model Constants
TRAINING_SIZE = 30
VALIDATION_SIZE = 10
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 500
LAYERS = 5
N_FEATURES = 3
N_CLASSES = 2  # is source/ not source
## GCNSI
ALPHA = 0.5  # controls the influence that a node gets from its neighbors in feature creation
GCNSI_N_FEATURES = 4
WEIGHT_DECAY = 0.1


# Graph Constants
MEAN_N_NODES = 50  # mean of number of nodes in the graph
## Watts-Strogatz
WS_NEIGHBOURS = 1.8  # control number of neighbours to connect to
WS_PROBABILITY = 0.4  # probability of rewiring an edge
## Barabasi-Albert
BA_NEIGHBOURS = 0.03  # control number of edges to attach between new and existing nodes


# Signal Propagation Constants
N_SOURCES = 2  # number of infection sources (probably make random later)
N_ITERATIONS = 40  # number of iterations for signal propagation
## SIR
SIR_BETA = 0.01  # probability for neighbor to get infected
SIR_GAMMA = 0.005  # probability of node to recover
## SI
SI_BETA = 0.01  # probability for neighbor to get infected

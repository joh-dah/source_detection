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
TRAINING_SIZE = 1000
VALIDATION_SIZE = 100
EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 10
LAYERS = 5
N_FEATURES = 3
N_CLASSES = 2  # is source/ not source
## GCNSI
GCNSI_N_FEATURES = 4
WEIGHT_DECAY = 0.1


# Graph Constants
N_NODES = 50
## Watts-Strogatz
WS_NEIGHBOURS = 5  # number of neighbors in ring topology to connect to
WS_PROBABILITY = 0.2  # probability of rewiring an edge

## Barabasi-Albert
BA_NEIGHBOURS = 3  # number of edges to attach from a new node to existing nodes

# Signal Propagation Constants
N_SOURCES = 1  # number of infection sources (probably make random later)
N_ITERATIONS = 40  # number of iterations for signal propagation
## SIR
SIR_BETA = 0.02  # probability for neighbor to get infected
SIR_GAMMA = 0.005  # probability of node to recover
## SI
SI_BETA = 0.01  # probability for neighbor to get infected

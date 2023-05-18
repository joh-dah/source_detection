# Run Parameters
PROP_MODEL = "SI"  # "SIR" or "SI"
GRAPH_TYPE = "watts_strogatz"  # "watts_strogatz" or "barabasi_albert"
MODEL = "GCN"  # "GCN"

# General
MODEL_PATH = "models"
FIGURES_PATH = "figures"
DATA_PATH = "data"
SEED = 123
N_FEATURES = 3  # is susceptible/infected/recovered at current status
N_CLASSES = 2  # is source/ not source

# Training Constants
EPOCHS = 30
LEARNING_RATE = 0.0005

# Graph Constants
N_NODES = 30
## Watts-Strogatz
WS_NEIGHBOURS = 8  # number of neighbors in ring topology to connect to
WS_PROBABILITY = 0.2  # probability of rewiring an edge

## Barabasi-Albert
BA_NEIGHBOURS = 3  # number of edges to attach from a new node to existing nodes

# Signal Propagation Constants
N_SOURCES = 1  # number of infection sources (probably make random later)
N_ITERATIONS = 60  # number of iterations for signal propagation
## SIR
SIR_BETA = 0.02  # probability for neighbor to get infected
SIR_GAMMA = 0.005  # probability of node to recover
## SI
SI_BETA = 0.01  # probability for neighbor to get infected

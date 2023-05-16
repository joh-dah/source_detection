# General
NET_PATH = "trained_nets"
FIGURES_PATH = "figures"
SEED = 123
N_FEATURES = 3  # is susceptible/infected/recovered at current status
N_CLASSES = 2  # is source/ not source

# Training Constants
EPOCHS = 30
LEARNING_RATE = 0.0005

# Graph Constants
N_NODES = 100

# Signal Propagation Constants
SIR_BETA = 0.02  # probability for neighbor to get infected
SIR_GAMMA = 0.005  # probability of node to recover
N_ITERATIONS = 60
N_SOURCES = 1  # number of infection sources (probably make random later)

# sourceDetection

## TODO 09.05-16.05

- Olli + Alle
    - GNN Paper
        - Was sind unsere Ziele?
        - Was unterscheidet das was wir machen wollen von diesem Paper?
    - GNN Tutorials

- Conrad
    - Signal Propagation
        - Welche Signal Propagation für welche Anwendungsfälle?
        - ...

- Sina
    - Graphen + Paper

- Johanna
    - Git aufräumen


## Goal

Benchmark performance of GNN's in Source Detection compared to unsupervised approaches

## Todo

### Step 1

- Research different Signal Propagation Models
    - Should be as realistic as possible
    - consider use cases like rumor spreading, computer viruses, biomedical stuff
    - check available python implementations

- Research different Graph representations of Networks
    - Should be as realistic as possible
    - consider use cases like rumor spreading, computer viruses, biomedical stuff
    - check available python implementations

### Step 2

- Create Python Framework for source detections via GNN's
    - implement interface to create synthetic data possibly based on different models/graphs
    - implement first version of GNN

- do some hyperparameter optimization on the GNN

### Step 3

- Use existing implementations of Unsupervised Methods for Source Detection to create Benchmarks

### Step 4

- Acquire real World Data from different use cases
    - Check if there is data with ground truth for source detection
    - otherwise use real world graphs and model signal propagation

- Benchmark GNN and unsupervised methods on real world data

### Step 5

- Write Report

# sourceDetection

## Setup (Linux approved)

Setup virtual environment
```
sudo apt install python3.10
sudo apt install python3-virtualenv

virtualenv -p python3.10 venv
source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

## Usage

```
python -m src.data_creation
python -m src.training
python -m src.validation
```

## ToDo

- Random input generation
- Signal Propagation und Graph erstellung: Parameter abhängig von größe des Graphens
- output softmax?
- Multiple Source Nodes --> Wie model ausgabe?
- Loss anders berechnen?
- (DVC) Pipeline bauen
- early stopping beim training
- GCNSI wie umgehen mit SIR model


## Link Collection
### GNN Tutorials:
- https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html
- https://towardsdatascience.com/graph-neural-networks-in-python-c310c7c18c83
- https://distill.pub/2021/gnn-intro/

### Types of real world graphs:
- https://aalab.cs.uni-kl.de/en/publikationen/peer_review/PDFs/KaufmannZweig_ModelingAndDesigningRealWorldNetworks.pdf
- https://noduslabs.com/radar/types-networks-random-small-world-scale-free/


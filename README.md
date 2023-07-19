# Source Detection with Graph Convolutional Networks

This repository provides a complete machine learning pipeline for the detection of sources in complex networks.

The pipeline consists of several key steps, including the generation of training data through the creation of random graphs and signal propagation. Additionally, it involves data preprocessing, training a graph convolutional network, validating the resulting model, and comparing its performance against existing unsupervised methods.

## Setup 

### Setup Virtual Environment (Linux):
```
sudo apt install python3.10
sudo apt install python3-virtualenv

virtualenv -p python3.10 venv
source venv/bin/activate
```

### Setup Virtual Environment (Windows):
Requirements: Python 3.10
```
python3.10 -m venv venv
venv\Scripts\activate
```
### Install Requirements
```
pip install -r requirements.txt
```

## Example-Script

The results of the validation will be written in a `json` file in the `reports` folder. Additionaly a roc curve will be generated in the `roc` folder. 
You can look in the documentation metioned below for a better understanding of the results.

## Usage (DVC)
This projects uses a dvc pipeline (https://dvc.org/).
To execute all stepts from the creation of the data, the processing, the training of the model and the evaluation you can execute `dvc repro`. 
The pipeline can be configured by the parameter in `params.yaml` and is defined in `dvc.yaml`.

## Documentation

You can find the documentation here.

## Recources

- For the implementation of unsupervised methods located in the `rpasdt` we used this repository: 
https://github.com/ElsevierSoftwareX/SOFTX-D-21-00167

- The data processing for the GCNSI follows the description in this paper: https://dl.acm.org/doi/pdf/10.1145/3357384.3357994
The arcitectures of the GCNSI and the GCNR are also inspired by this paper.
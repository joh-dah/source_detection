# Data Processing Documentation

This documentation provides a detailed overview of the data processing pipeline used to create processed data based on the selected model. It covers the preprocessing steps, dataset creation, and the specific processing functions for different models. It also discusses the two different labeling approaches used in the data processing.

## Preprocessing Steps
Before creating the processed data, the following preprocessing steps are performed:

### Data Loading
The input data is loaded from the saved datasets, whether synthetic or real-world. The data includes the graph structure, node features, and ground truth labels.

### Train-Validation Split
The dataset is split into training and validation sets. The `--validation` flag is used to determine whether the processed data is created for validation or training.

## Dataset Creation
The processed data is represented as an instance of the SDDataset class, which is a subclass of `torch_geometric.data.Dataset`. This class encapsulates the graph structure, node features, and ground truth labels, and provides convenient methods for accessing and manipulating the data.

The SDDataset class includes the following important methods:

`process()`: This method is responsible for processing the raw data and creating the processed data. It applies the specified pre_transform function to each raw data instance and saves the processed data as separate files.

`get()`: This method retrieves a specific data instance from the processed dataset based on the provided index.

## Labeling Approaches
The data processing pipeline supports two different labeling approaches:

### Binary Classification (GCNSI)
This approach labels the nodes as either a source or a non-source. It transforms the ground truth labels into a binary classification problem, where nodes that are sources have a positive label (1) and nodes that are not sources have a negative label (0).

### Regression (GCNR)
This approach treats the problem as a regression task, where the labels represent the distance of each node to the nearest source. 

## Processing Functions
The processing functions are used to transform the raw data into the desired format for the selected model. There are three processing functions defined:

`process_gcnsi_data`: This function prepares the features and labels for the GCNSI approach. It uses the `paper_input` function to transform the node features according to the input generation in the paper “Multiple Rumor Source Detection with Graph Convolutional Networks”. The labels are transformed into a binary classification problem, where the nodes are labeled as either a source or a non-source. The labels are expanded to a 2D tensor, and the processed data is returned.

`process_simplified_gcnsi_data`: This function provides a simplified version of the features and labels for the GCNSI approach. It expands the features to a 2D tensor and keeps the labels as a binary classification problem, similar to the process_gcnsi_data function.

`process_gcnr_data`: This function prepares the features and labels for the GCNR approach. It uses the `paper_input` function to transform the node features. The labels are created as distance labels, where each label represents the distance of a node to the nearest source. The distance labels are created using the `create_distance_labels function`, which calculates the distance of each node to the nearest source. The processed data is returned with expanded features and distance labels.

## Data Saving
The processed data is stored in the processed directory within the dataset directory in `data`.

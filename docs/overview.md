# Documentation

The pipeline of our framework consists of several stages, each contributing to the overall process of source detection with Graph Convolutional Networks (GCNs). In this documentation, we will provide an overview of each stage.

## Stages of the Pipeline

### 1. Data Creation
This stage involves the generation of a dataset comprising graphs with modeled signal propagation. Random graphs are created, and signal propagation simulations are performed on these random graphs and on real-world graphs using an SI model. The resulting data points include the graph information and the signal statuses of nodes.

You can find more information [here](https://github.com/joh-dah/source_detection/blob/main/docs/data_creation.md).
### 2. Data Processing
The data processing stage is responsible for preparing the dataset generated in the data creation stage for training and validation. This involves applying transformations and preprocessing steps to the raw data to enhance the performance of the Graph Convolutional Networks (GCNs) used for source detection.

You can find more information [here](https://github.com/joh-dah/source_detection/blob/main/docs/data_processing.md).
### 3. Training
The training stage is responsible for training the specified classifier using the given dataset. This stage utilizes the selected model architecture and loss criterion to optimize the model parameters based on the training data.

You can find more information [here](https://github.com/joh-dah/source_detection/blob/main/docs/training.md).
### 4. Validation
The validation stage is responsible for evaluating the performance of the generated model. This stage involves computing various evaluation metrics to assess the model's accuracy and effectiveness.

You can find more information [here](https://github.com/joh-dah/source_detection/blob/main/docs/validation.md).

## Parameters
The parameters of each stage can be set in the `params.yaml`.
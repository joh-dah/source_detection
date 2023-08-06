# Training Documentation

This documentation provides an overview of the training process for the model with the specified parameters. It covers the model initialization, loss functions, data subsampling, parallel training, and model saving.  

## Model Initialization
The training process begins by initializing the specified labeling approach based on the `const.MODEL` value. Currently, two types are supported:

### GCNR (Graph Convolutional Network for Regression)
This model is initialized using the GCNR class from the `architectures.GCNR` module. It is designed for regression tasks and predicting the distance of each node to the nearest source.

### GCNSI (Graph Convolutional Network based Source Identification)
This model is initialized using the GCNSI class from the `architectures.GCNSI` module. It is designed for binary classification tasks and identifying nodes as either sources or non-sources.

## Loss Functions
The choice of the loss function depends on the selected labeling approach. The following loss functions are used:

### Loss Functions for GCNR
The model uses the Mean Squared Logarithmic Error (MSLE) loss function defined in the `MSLELoss` class. It measures the mean squared logarithmic difference between the predicted and actual distances. Alternatively, if the parameter `const.USE_LOG_LOSS` is `False`, the model uses the Mean Squared Error (MSE) loss function.

### Loss Functions for GCNSI
The model uses the Binary Cross Entropy with Logits (BCEWithLogitsLoss) loss function from `torch.nn module`. It is suitable for binary classification problems where the model predicts the probability of each node being a source.

## Data Subsampling
Data subsampling is an optional step that can be enabled by setting `const.SUBSAMPLE` to `True`. It helps address class imbalance by randomly subsampling the majority class (non-sources) to balance it with the minority class (sources). The subsampleClasses function performs the subsampling process, ensuring an equal number of samples from each class.

## Parallel Training
If multiple GPUs are available (`torch.cuda.device_count() > 1`), the training process utilizes all available GPUs for parallel training. The `DataParallel` class from `torch_geometric.nn.data_parallel` is used to parallelize the model training.

## Model Saving
During training, the model with the best validation loss is saved in `models`. The model is saved both as the "latest" model and with a unique name based on the current date and time. 
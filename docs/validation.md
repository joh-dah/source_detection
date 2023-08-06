# Validation Documentation
The validation module is responsible for evaluating the performance of the trained model on a validation dataset. It calculates various metrics to assess the accuracy and effectiveness of the model's predictions. Here are the details of the validation process:

## Configuration
The validation process uses the classifier specified in the `params.yaml`. The dataset for the validation is specified using the `--dataset` command-line argument.

## Metrics
`ROC AUC Score`: The Area Under the Receiver Operating Characteristic Curve (ROC AUC) measures the model's ability to discriminate between positive and negative instances.

`F1 Score`: F1 score combines precision and recall to evaluate the model's overall performance.

`Minimum Matching Distance`: This metric measures the average minimum matching distance between the true sources and the predicted sources. It quantifies the accuracy of the predicted sources.

`Rank`: This metric computes the average rank of the true sources among the predicted sources. It indicates how well the model ranks the true sources in terms of their likelihood.

## Unsupervised Metrics
In addition to our supervised approach, the validation module also evaluates an unsupervised method called NETSLEUTH on the validation dataset.
NETSLEUTH is an unsupervised source detection algorithm implemented in the repository https://github.com/ElsevierSoftwareX/SOFTX-D-21-00167 and located in `rpastd`.
This allows us to compare the performance of our trained model with the performance of NETSLEUTH. The metrics used for comparison include the average minimum matching distance and the average F1 score.

## Data Statistics
The validation module calculates various graph-related and infection-related statistics for the validation dataset. The graph-related statistics include the average number of nodes and average centrality of the graphs. The infection-related statistics include the average number of sources, average portion of infected nodes, and standard deviation of the portion of infected nodes.

## Saving Results
The validation results, including the computed metrics, data statistics, and model parameters, are saved in a file in the `reports` dictionary. The generated ROC-Curves are saved in the `roc` dictionary.
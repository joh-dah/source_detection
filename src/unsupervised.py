import argparse
import yaml
from src import utils
import src.validation as val
import datetime


def main():
    """
    Initiates the validation of the classifier specified in the constants file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="synthetic", help="name of the dataset"
    )
    args = parser.parse_args()

    dataset = args.dataset.lower()
    print(f"Load Dataset: {dataset}")
    raw_val_data = utils.load_raw_data(dataset, True)

    metrics_dict = {}
    metrics_dict["dataset"] = dataset
    metrics_dict["metrics"] = val.unsupervised_metrics(raw_val_data)
    metrics_dict["data stats"] = val.data_stats(raw_val_data)
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))

    model_name = "unsup_" + datetime.datetime.now().strftime("%m-%d_%H-%M")
    utils.save_metrics(metrics_dict, model_name, dataset)


if __name__ == "__main__":
    main()

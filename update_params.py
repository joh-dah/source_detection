import argparse
import json


def update_params(params, change_dict, idx):
    for key, value in change_dict.items():
        if isinstance(value, dict):
            update_params(params[key], value, idx)
        else:
            print(f"Updating {key} to {value[idx]}")
            params[key] = value[idx]


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="Slurm job array index")
args = parser.parse_args()

change_dict = {
    "model": ["GCNSI", "GCNSI", "GCNR", "GCNR", "GCNR", "GCNSI", "GCNR"],
    "training": {"epochs": [30, 30, 100, 100, 100, 100, 100],
                 "learning_rate": [0.001, 0.0001, 0.001, 0.0001, 0.00001, 0.00001, 0.00005],},
    "data_creation": {"mean_nodes": [1000, 1000, 1000, 1000, 1000, 1000, 1000],
                      "train_size": [1000, 1000, 1000, 1000, 10000, 10000, 10000],},
}

# load the params.json file
with open("params.json") as json_file:
    params = json.load(json_file)

update_params(params, change_dict, args.idx)
params["model_name"] = f"{args.idx}"

# save the updated params.json file
with open("params.json", "w") as outfile:
    json.dump(params, outfile, indent=4)

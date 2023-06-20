import argparse
import json


def update_params(params: dict, change_dict, idx):
    for key, value in change_dict.items():
        if isinstance(value, dict):
            update_params(params[key], value, idx)
        else:
            print(f"Updating {key} to {value[idx]}")
            assert key in params
            params[key] = value[idx]


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="Slurm job array index")
args = parser.parse_args()

change_dict = {
    "model": ["GCNSI", "GCNR"],
    "training": {
        "epochs": [100, 100],
        "learning_rate": [0.00001, 0.00001],
        "useLogLoss": [False, True],
    },
    "data_creation": {
        "mean_nodes": [1000, 1000],
        "training_size": [10000, 10000],
    },
}

# load the params.json file
with open("params.json") as json_file:
    params = json.load(json_file)

update_params(params, change_dict, args.idx)
params["model_name"] = f"{args.idx}"
params["on_cluster"] = True

# save the updated params.json file
with open("params.json", "w") as outfile:
    json.dump(params, outfile, indent=4)

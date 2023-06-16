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

change_dict = {"model": ["GCNSI", "GCNR"], "training": {"epochs": [100, 200]}}

# load the params.json file
with open("params.json") as json_file:
    params = json.load(json_file)

update_params(params, change_dict, args.idx)

# save the updated params.json file
with open("params.json", "w") as outfile:
    json.dump(params, outfile, indent=4)

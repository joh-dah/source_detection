import argparse
import yaml


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
    "model": ["GCNSI", "GCNR", "GCNR"],
    "GCNR_layer_type": ["GCN", "GCN", "GAT"],
    "training": {
        "layers": [10, 10, 4],
        "epochs": [100, 100, 150],
        "batch_size": [30, 30, 20],
        "hidden_size": [512, 512, 300],
    },
}

# load the params.json file
with open("params.yaml") as yaml_file:
    params = yaml.full_load(yaml_file)

update_params(params, change_dict, args.idx)

# save the updated params.json file
with open("params.yaml", "w") as outfile:
    yaml.dump(params, outfile, indent=4)

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
    "training": {
        "useLogLoss": [False],
    }
}

# load the params.json file
with open("params.json") as yaml_file:
    params = yaml.full_load(yaml_file)

update_params(params, change_dict, args.idx)

# save the updated params.json file
with open("params.json", "w") as outfile:
    yaml.dump(params, outfile, indent=4)

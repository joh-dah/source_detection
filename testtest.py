import yaml
import json

# read params.json
with open("params.json", "r") as json_file:
    params = json.load(json_file)

# write params.yaml
with open("params_n.yaml", "w") as yaml_file:
    params_yaml = yaml.dump(params, yaml_file)

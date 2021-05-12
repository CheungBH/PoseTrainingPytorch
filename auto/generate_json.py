import json
import sys
import os
from typing import List

lists = {"data_path":"/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/PoseTrainingPytorch/data_cfg1.json",
         'model_path':'/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/PoseTrainingPytorch/model_cfg1.json',
         "kps": 13, "backbone":"seresnet18", "loadModel": "model.pth", "sigma": 4}


data_cfg_path = "../config/data_cfg/data_default.json"
model_cfg_path = "../config/model_cfg/default/cfg_resnet18.json"


def generate_json():
    args = lists
    dest_data_path, dest_cfg_path = args["data_path"], args['model_path']
    res = {}
    res1 = {}
    data_default = json.load(open(data_cfg_path))
    model_default = json.load(open(model_cfg_path))
    for key in data_default:
        if key in args:
            if data_default[key] != args[key]:
                res[key] = args[key]
        else:
            res[key] = data_default[key]
    for keys in model_default:
        if keys in args:
            if model_default[keys] != args[keys]:
                res1[keys] = args[keys]
        else:
            res1[keys] = model_default[keys]
    json_out = open(dest_data_path, "w")
    json_out1 = open(dest_cfg_path,'w')
    json_out.write(json.dumps(res))
    json_out1.write(json.dumps(res1))


if __name__ == '__main__':
    generate_json()


# "python generate_json.py data_cfg1.json model_cfg1.json --kps 13 --backbone seresnet18 --loadModel model.pth --sigma 4"
# "python generate_json.py data_cfg2.json model_cfg2.json --LR 1E-3 --backbone seresnet101 --sigma 2 --input_height 256 --input_width 256 --output_height 64 --output_width 64"
# "python generate_json.py data_cfg3.json model_cfg3.json --LR 1E-3 --backbone seresnet101 --sigma 1 --input_height 320 --input_width 320 --output_height 80 --output_width 80 --loadModel model.pth --se_ratio -1 --scale 0.3 --optMethod adam"

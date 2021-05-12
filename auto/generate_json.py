import json
import sys
import os
from typing import List


data_cfg_path = "../config/data_cfg/data_default.json"


def generate_json():
    s = sys.argv
    n = len(s)
    a,b = [],[]
    for i in range(3,n):
        if i % 2 != 0:
            a.append(s[i])
        else:b.append(s[i])
    args = dict(zip(a,b))
    dest_data_path, dest_cfg_path = s[1], s[2]
    res,res1 = {},{}
    model_json = select_model_cfg(args['backbone'])
    data_default = json.load(open(data_cfg_path))
    model_default = json.load(open(model_json))
    for key in data_default:
        if key in args:
            if data_default[key] != args[key]:
                res[key] = args[key]
        else:
            res[key] = data_default[key]
    select_model_cfg(args['backbone'])
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

def select_model_cfg(backbone):
    # if backbone == "mobilenet":
    #     return "../config/model_cfg/default/cfg_mobile.json"
    if backbone == "seresnet18":
        return "../config/model_cfg/default/cfg_resnet18.json"
    elif backbone == "seresnet50":
        return "../config/model_cfg/default/cfg_seresnet50.json"
    elif backbone == "seresnet101":
        return "../config/model_cfg/default/cfg_resnet101.json"
    elif backbone == "shufflenet":
        return "../config/model_cfg/default/cfg_shuffle.json"
    else:
        raise NotImplementedError


if __name__ == '__main__':
    generate_json()

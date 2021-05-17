import json
import sys
import os

data_cfg_path = "config/data_cfg/data_default.json"


def generate_json():
    s = sys.argv
    n = len(s)
    a,b,nums = [],[],[]
    for i in range(3,n):
        if s[i] != '--freeze_bn':nums.append(s[i])
        else:continue
    for j in range(len(nums)):
        if j % 2 == 0:a.append(nums[j])
        else: b.append(nums[j])
    args = dict(zip(a,b))
    dest_data_path, dest_cfg_path = s[1], s[2]
    res,res1 = {},{}
    model_json = select_model_cfg(args['--backbone'])
    data_default = json.load(open(data_cfg_path))
    model_default = json.load(open(model_json))
    for key in data_default:
        key_y = os.path.join("--" + key)
        if key_y in args:
            if data_default[key] != args[key_y]:
                if isinstance(data_default[key], int):
                    res[key] = int(args[key_y])
                elif isinstance(data_default[key], float):
                    res[key] = float(args[key_y])
                else:
                    res[key] = args[key_y]
        else:
            res[key] = data_default[key]
    select_model_cfg(args['--backbone'])
    for keys in model_default:
        key_y = os.path.join("--" + key)
        if key_y in args:
            if model_default[keys] != args[key_y]:
                if isinstance(model_default[keys], int):
                    res1[keys] = int(args[key_y])
                elif isinstance(model_default[keys], float):
                    res1[keys] = float(args[key_y])
                else:
                    res1[keys] = args[key_y]
        else:
            res1[keys] = model_default[keys]
    if len(nums) == n-4:
        res1['--freeze_bn'] = None
    json_out = open(dest_data_path, "w")
    json_out1 = open(dest_cfg_path,'w')
    json_out.write(json.dumps(res))
    json_out1.write(json.dumps(res1))


def select_model_cfg(backbone):
    if backbone == "seresnet18":
        return "config/model_cfg/default/cfg_resnet18.json"
    elif backbone == "seresnet50":
        return "config/model_cfg/default/cfg_seresnet50.json"
    elif backbone == "seresnet101":
        return "config/model_cfg/default/cfg_resnet101.json"
    elif backbone == "shufflenet":
        return "config/model_cfg/default/cfg_shuffle.json"
    elif backbone == "mobilenet":
        return "config/model_cfg/default/cfg_mobile.json"
    else:
        raise NotImplementedError


if __name__ == '__main__':
    generate_json()


"python generate_json.py data_cfg1.json model_cfg1.json --kps 13 --backbone seresnet18 --loadModel model.pth --sigma 4 --freeze_bn"
"python generate_json.py data_cfg2.json model_cfg2.json --LR 1E-3 --backbone seresnet101 --sigma 2  --freeze_bn --input_height 256 --input_width 256 --output_height 64 --output_width 64"
"python generate_json.py data_cfg3.json model_cfg3.json --LR 1E-3 --backbone seresnet101 --sigma 1 --input_height 320 --input_width 320 --output_height 80 --output_width 80 --loadModel model.pth --se_ratio -1 --scale 0.3 --optMethod adam"
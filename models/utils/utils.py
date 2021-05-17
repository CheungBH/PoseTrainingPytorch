import json


def write_cfg(res, name2json):
    Obj = json.dumps(res)
    Object = open(name2json, 'w')
    Object.write(Obj)
    Object.close()
    return name2json


def parse_cfg(imgdir):
    with open(imgdir, "r") as load_f:
        load_dict = json.load(load_f)
    result = {
        "backbone": load_dict["backbone"],
        "residual": load_dict["residual"],
        "channels": load_dict["channels"],
        "kps": load_dict["kps"],
        "se_ratio": load_dict["se_ratio"],
        "head_type": load_dict["head_type"],
        "head_channel": load_dict["head_channel"],
        "first_conv": load_dict["first_conv"]
    }
    return result


def parse_DUC(cfg):
    return cfg["head_channel"], cfg["kps"]


def parse_se(cfg):
    se = cfg["se_ratio"]
    if se > 0:
        return abs(se), True
    else:
        return abs(se), False


def parse_num_block(cfg):
    try:
        channels = cfg["channels"].values()
        return [len(channel) for channel in channels]
    except:
        return 0


def parse_channel(cfg):
    try:
        channels = cfg["channels"].values()
        return [i for sub_ls in channels for sub_itm in sub_ls for i in sub_itm]
    except:
        return 0


if __name__ == '__main__':
    data_cfg = "../../dataset/data_default.json"
    dicc = {"input_height": 256,
            "input_width": 256,
            "output_height": 64,
            "output_width": 64,
            "sigma": 1,
            "rotate": 40,
            "flip_prob": 0.5,
            "scale": 0.2,
            "kps": 13
            }
    write_cfg(dicc, data_cfg)

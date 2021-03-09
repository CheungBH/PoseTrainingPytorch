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
        "keypoints": load_dict["keypoints"],
        "se_ratio": load_dict["se_ratio"],
        "head_type": load_dict["head_type"],
        "head_channel": load_dict["head_channel"],
        "first_conv": load_dict["first_conv"]
    }
    return result


def parse_DUC(cfg):
    return cfg["head_channel"], cfg["keypoints"]


def parse_se(cfg):
    se = cfg["se_ratio"]
    if se > 0:
        return abs(se), True
    else:
        return abs(se), False


def parse_num_block(cfg):
    channels = cfg["channels"].values()
    return [len(channel) for channel in channels]


def parse_channel(cfg):
    channels = cfg["channels"].values()
    return [i for sub_ls in channels for sub_itm in sub_ls for i in sub_itm]

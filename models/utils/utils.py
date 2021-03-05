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
        "keyponits": load_dict["keyponits"],
        "se_ratio": load_dict["se_ratio"],
        "head_type": load_dict["head_type"],
        "head_channel": load_dict["head_channel"]
    }
    return result
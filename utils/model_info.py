import json
import os
res = {'backbone':'seresnet18','keyponits':17,'se_ratio':16,'residual':[100,200,300,400],'channels':{1: [[61], [62]],2: [[13], [15]],3: [[9], [44]],4: [[16], [122]]},"head_type": "pixelshuffle","head_channel":[248,124]}


# res = {"1":{'backbone':'seresnet18','keyponits':17,'se_ratio':16,'residual':[100,200,300,400],'channels':{1: [[61], [62]],2: [[13], [15]],3: [[9], [44]],4: [[16], [122]]},"head":[248,124]},
#        '2':{'backbone':'seresnet50','keyponits':17,'se_ratio':16,'residual':[100,200,300,400],'channels':{1: [[61], [62]],2: [[13], [15]],3: [[9], [44]],4: [[16], [122]]},"head":[248,124]}}

def write_cfg(res):
    Obj = json.dumps(res)
    name2json = 'jsonFile.json'
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

if __name__ == '__main__':
    json_name = write_cfg(res)
    dest_folder = ''
    img_dir = os.path.join(dest_folder, json_name)
    res = parse_cfg(img_dir)
    print(res)

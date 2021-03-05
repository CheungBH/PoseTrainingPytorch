import json
import numpy as np


# res = {'1':{'backbone':'seresnet18','keyponits':17,'se_ratio':16,'residual':[100,200,300,400],'channels':{1: [[61], [62]],2: [[13], [15]],3: [[9], [44]],4: [[16], [122]]},"head":[248,124]},
#        '2':{'backbone':'seresnet50','keyponits':17,'se_ratio':16,'residual':[100,200,300,400],'channels':{1: [[61], [62]],2: [[13], [15]],3: [[9], [44]],4: [[16], [122]]},"head":[248,124]}}
# jsObj = json.dumps(res)
#
# fileObject = open('jsonFile.json', 'w')
# fileObject.write(jsObj)
# fileObject.close()
# #
# backbone = "seresnet18"
# keypoint = 17
# se_ratio = 16
# residual = [100, 200, 300, 400]
# channels = {1: [[61], [62]],
# 2: [[13], [15]],
# 3: [[9], [44]],
# 4: [[16], [122]]
# }
# head = [248, 124]
#
#
# backbone = "seresnet50"
# keypoint = 13
# se_ratio = -1
# residual = [100, 200, 300, 400]
# channels = {1: [[61, 12], [62, 71], [55, 52]],
# 2: [[13, 12], [15, 87], [44, 87], [99, 6]],
# 3: [[61, 12], [62, 71], [55, 52], [61, 12], [62, 71], [55, 52]],
# 4: [[62, 71], [55, 52], [44, 87]]
# }
# head = [168, 124, 88]
def write_cfg(res):
    jsObj = json.dumps(res)
    fileObject = open('jsonFile.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()

def read_cfg(imgdir):
    res = {}
    with open(imgdir, "r") as load_f:
        load_dict = json.load(load_f)
        residual = {}
        head = {}
        se_ratio = {}
        backbone = {}
        channels = {}
    for i in load_dict.keys():
        residual[i] = load_dict[i]["residual"]
        head[i] = load_dict[i]["head"]
        se_ratio[i] = load_dict[i]['se_ratio']
        backbone[i] = load_dict[i]["backbone"]
        channels[i] = load_dict[i]["channels"]
    # for index in backbone.keys():
    #     if index in residual.keys() and index in head.keys() and index in se_ratio.keys() and index in channels.keys():
    #         res['residual'] = {
    #             'residual' :load_dict[i]["residual"],
    #                'head': load_dict[i]["head"],
    #                'se_ratio': load_dict[i]["se_ratio"],
    #                'backbone': load_dict[i]["backbone"],
    #                'keyponits': load_dict[i]["keyponits"]
    #         }
    return res

if __name__ == '__main__':
    img_dir = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Autoannotation_Pose/Img2json/jsonFile.json"
    read_cfg(img_dir)

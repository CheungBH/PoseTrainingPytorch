#-*-coding:utf-8-*-
import json

json_paths = [
    "/home/hkuit155/Documents/pose_track/lighttrack/data/Data_2018/posetrack_data/annotations/val/"
    "000342_mpii_test.json",
    "/home/hkuit155/Documents/mmpose/tools/work_dirs/res50_posetrack18_256x192/preds/000342_mpii_test.json"
]
load_dicts = []
for json_path in json_paths:
    with open(json_path, "r") as load_f:
        load_dict = json.load(load_f)
    load_dicts.append(load_dict)

a = 1

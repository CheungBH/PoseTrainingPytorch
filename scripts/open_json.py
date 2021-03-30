import json

json_path = r"E:\mmpose\work_dirs\res50_posetrack18_256x192\preds\023752_mpii_test.json"

with open(json_path, "r") as load_f:
    load_dict = json.load(load_f)

print(load_dict)

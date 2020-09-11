from test import main
import os
from src.opt import opt
import torch
import csv
from utils.utils import write_test_title
from config.config import computer

model_folders = "test_weight"
test_data = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}


result_path = "result.csv"
if_exist = os.path.exists(result_path)
test_log = open(result_path, "a+", newline="")
csv_writer = csv.writer(test_log)
if not if_exist:
    csv_writer.writerow(write_test_title())

for folder in os.listdir(model_folders):
    option, model, test_log = "", "", []
    for file in os.listdir(os.path.join(model_folders, folder)):
        if "option" in file:
            option = os.path.join(model_folders, folder, file)
        elif ".pkl" in file or ".pth" in file:
            model = os.path.join(model_folders, folder, file)
            test_log.append(folder+model)
        else:
            continue

    assert option != "" and model != ""

    info = torch.load(option)
    cfg = info.struct
    backbone = info.backbone
    opt.kps = info.kps

    benchmark, overall, part = main(backbone, cfg, test_data, model)

    for item in benchmark:
        test_log.append(item)
    test_log.append(computer)

    for item in overall:
        test_log.append(item)
    test_log.append(" ")

    for indicator in part:
        for kp in indicator:
            test_log.append(kp)
        test_log.append(" ")

    csv_writer.writerow(test_log)



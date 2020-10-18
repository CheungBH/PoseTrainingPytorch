from test import main
import os
from src.opt import opt
import torch
import csv
from utils.utils import write_test_title
from config.config import computer

model_folders = "exp/coco_mob"
test_data = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}
test_mark = ["best_acc", "best_auc", "best_pr", "best_dist"]

result_path = os.path.join(model_folders, "test_result.csv")
if_exist = os.path.exists(result_path)
test_log = open(result_path, "a+", newline="")

model_ls, option_ls, name_ls, folder_ls = [], [], [], []
csv_writer = csv.writer(test_log)

for folder in os.listdir(model_folders):
    if not os.path.isdir(os.path.join(model_folders, folder)):
        continue

    option, model = "", ""
    model_tmp, name_tmp = [], []
    if "csv" in folder:
        continue

    for file in os.listdir(os.path.join(model_folders, folder)):
        if "option" in file:
            option = os.path.join(model_folders, folder, file)
        elif ".pkl" in file or ".pth" in file:
            for mark in test_mark:
                if mark in file:
                    model = os.path.join(model_folders, folder, file)
                    model_tmp.append(model)
                    name_tmp.append(mark)
        else:
            continue

    assert option != "" and model != ""

    for m, n in zip(model_tmp, name_tmp):
        model_ls.append(m)
        option_ls.append(option)
        name_ls.append(n)
        folder_ls.append(folder)

info = torch.load(option)
opt.kps = info.kps

if not if_exist:
    csv_writer.writerow(write_test_title())

for op, mdl, name, fold in zip(option_ls, model_ls, name_ls, folder_ls):
    log = [fold, name]

    info = torch.load(op)
    cfg = info.struct
    backbone = info.backbone
    opt.kps = info.kps
    opt.DUC = info.DUC

    print("Testing model {}".format(mdl))
    benchmark, overall, part = main(backbone, cfg, test_data, mdl)

    for item in benchmark:
        log.append(item)
    log.append(computer)

    for item in overall:
        if isinstance(item, torch.Tensor):
            item = item.tolist()
        log.append(item)
    log.append(" ")

    for indicator in part:
        for kp in indicator:
            if isinstance(kp, torch.Tensor):
                kp = kp.tolist()
            log.append(kp)
        log.append(" ")

    csv_writer.writerow(log)



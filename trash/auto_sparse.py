from sparse import detect_sparse
import os
from config.opt import opt
import torch

model_folders = "test_weight/ceiling_0911"
result_path = os.path.join(model_folders, "sparse_result.csv")


for folder in os.listdir(model_folders):
    option, model, test_log = "", "", []
    if "csv" in folder:
        continue

    for file in os.listdir(os.path.join(model_folders, folder)):
        if "option" in file:
            option = os.path.join(model_folders, folder, file)
        elif ".pkl" in file or ".pth" in file:
            model = os.path.join(model_folders, folder, file)
        else:
            continue

    assert option != "" and model != ""

    info = torch.load(option)
    opt.struct = info.struct
    opt.backbone = info.backbone
    opt.kps = info.kps
    opt.DUC = info.DUC

    print("Detecting sparsity of model {}".format(model))
    detect_sparse(model, result_path)




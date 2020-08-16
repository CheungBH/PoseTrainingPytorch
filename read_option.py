#-*-coding:utf-8-*-
import torch

option_path = "exp/aic_origin/163/option.pkl"
info = torch.load(option_path)
print(info)
print(info.kps)
print(info.backbone)
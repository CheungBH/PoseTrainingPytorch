#-*-coding:utf-8-*-
import torch

option_path = "exp/underwater-mobile_13kps/434/option.pkl"
info = torch.load(option_path)
print(info)
print(info.kps)
print(info.backbone)
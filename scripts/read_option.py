#-*-coding:utf-8-*-
import torch

option_path = "../exp/test/adam/option.pkl"
info = torch.load(option_path)
print(info)
# print(info.kps)
# print(info.backbone)
# print(info.se_ratio)
# a = 1

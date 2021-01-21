#-*-coding:utf-8-*-
import torch

option_path = "D:/pose_test1/10/option.pkl"
info = torch.load(option_path)
print(info)
print(info.kps)
print(info.backbone)
print(info.thresh)
a = 1

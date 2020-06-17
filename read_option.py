#-*-coding:utf-8-*-
import torch

option_path = "exp/coco/efficientnet-b3/option.pkl"
info = torch.load(option_path)
print(info)

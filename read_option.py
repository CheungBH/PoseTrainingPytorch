#-*-coding:utf-8-*-
import torch

option_path = "exp/aic_origin/16/option.pkl"
info = torch.load(option_path)
print(info)

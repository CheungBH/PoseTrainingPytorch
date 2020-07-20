#-*-coding:utf-8-*-
import torch

option_path = "exp/aic_0720/seres101_origin/option.pkl"
info = torch.load(option_path)
print(info)

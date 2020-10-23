#-*-coding:utf-8-*-
import torch

option_path = "test_weight/ceiling_thresh/78/option.pkl"
info = torch.load(option_path)
print(info)
# print(info.kps)
# print(info.backbone)
# print(info.thresh)
a = 1

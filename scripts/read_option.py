#-*-coding:utf-8-*-
import torch

option_path = "/home/hkuit164/Desktop/pose/PoseTrainingPytorch_1/exp/tennis_player_64/mob_bs8_0.001/option.pkl"
info = torch.load(option_path)
print(info)
# print(info.kps)
# print(info.backbone)
# print(info.se_ratio)
# a = 1

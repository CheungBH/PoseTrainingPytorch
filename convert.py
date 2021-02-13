from models.pose_model import PoseModel
from utils.utils import get_option_path
import torch
import os

model_path = ""
onnx = ""
libtorch = ""

'''Must be assigned if option.pkl is not exist'''
backbone = ""
cfg = ""
height = ""
width = ""

option_path = get_option_path(model_path)
if os.path.join(option_path):
    option = torch.load(option_path)
    backbone = option.backbone
    cfg = option.struct
    height = option.inputResH
    width = option.inputResW

posenet = PoseModel()
posenet.build(backbone, cfg)
posenet.load(model_path)
model = posenet.model

with torch.no_grad():
    dummy_inp = torch.ones()
    if onnx:


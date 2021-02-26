from models.pose_model import PoseModel
from utils.utils import get_option_path
import torch
import os
from src.opt import opt

model_path = "onnx/models_bh_0226/seresnet18/aic_original_best_acc.pkl"
cfg = ""
onnx_path = "onnx/models_bh_0226/seresnet18/model.onnx"
libtorch_path = "onnx/models_bh_0226/seresnet18/model.pt"

'''Must be assigned if option.pkl is not exist'''
backbone = ""
height = ""
width = ""
opt.kps = 13
opt.se_ratio = 16

option_path = get_option_path(model_path)
if os.path.join(option_path):
    option = torch.load(option_path)
    backbone = option.backbone
    # cfg = option.struct
    height = option.inputResH
    width = option.inputResW
    opt.kps = option.kps
    try:
        opt.se_ratio = option.se_ratio
    except:
        opt.se_ratio = 1

posenet = PoseModel()
posenet.build(backbone, cfg)
posenet.load(model_path)
model = posenet.model

example = torch.rand(2, 3, width, height).cuda()
with torch.no_grad():
    if libtorch_path:
        traced_model = torch.jit.trace(model, example)
        traced_model.save(libtorch_path)
    if onnx_path:
        torch_out = torch.onnx.export(model, example, onnx_path, verbose=False, )

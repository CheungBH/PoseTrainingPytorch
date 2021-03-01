from models.pose_model import PoseModel
from utils.utils import get_option_path
import torch
import os
from src.opt import opt


class Converter:
    '''Must be assigned if option.pkl is not exist'''
    backbone = "seresnet50"
    height = 320
    width = 256
    opt.kps = 17
    opt.se_ratio = 16

    def __init__(self, model_path, cfg=None, onnx_path="model.onnx", libtorch_path="model.pt", onnx_sim_path="model_sim.onnx", device="cpu"):
        self.src_model_path = model_path
        self.onnx_path = onnx_path
        self.libtorch_path = libtorch_path
        self.onnx_sim_path = onnx_sim_path
        self.model_cfg = cfg
        self.device = device
        self.superior_path = "/".join(model_path.replace("\\", "/").split("/")[:-1])
        self.load_options()
        self.load_model()
        self.dummy_input = torch.rand(2, 3, self.width, self.height)
        if self.device == "cpu":
            self.dummy_input = torch.rand(2, 3, self.width, self.height).cuda()

    def onnx_convert(self):
        if self.onnx_path:
            torch_out = torch.onnx.export(self.model, self.dummy_input, self.onnx_path, verbose=False, )
            if self.onnx_sim_path:
                import os
                os.system("python -m onnxsim {} {}".format(self.onnx_path, self.onnx_sim_path))

    def libtorch_convert(self):
        if self.libtorch_path:
            traced_model = torch.jit.trace(self.model, self.dummy_input)
            traced_model.save(self.libtorch_path)

    def load_model(self):
        posenet = PoseModel()
        posenet.build(self.backbone, self.model_cfg)
        posenet.load(self.src_model_path)
        self.model = posenet.model

    def load_options(self):
        option_path = get_option_path(self.src_model_path)
        if os.path.exists(option_path):
            option = torch.load(option_path)
            self.backbone = option.backbone
            self.height = option.inputResH
            self.width = option.inputResW
            opt.kps = option.kps
            try:
                opt.se_ratio = option.se_ratio
            except:
                opt.se_ratio = 1

    def convert(self):
        with torch.no_grad():
            self.onnx_convert()
            self.libtorch_convert()


if __name__ == '__main__':
    pytorch_model = "pruned_seresnet50.pth"
    cfg = "cfg_seresnet50.txt"
    convert = Converter(pytorch_model, cfg=cfg)
    convert.convert()

'''
model_path = "onnx/models_bh_0226/seresnet18/aic_original_best_acc.pkl"
cfg = ""
onnx_path = "onnx/models_bh_0226/seresnet18/model.onnx"
libtorch_path = "onnx/models_bh_0226/seresnet18/model.pt"

# Must be assigned if option.pkl is not exist
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

'''

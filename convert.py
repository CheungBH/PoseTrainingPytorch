from models.pose_model import PoseModel
from utils.utils import get_option_path, get_corresponding_cfg
import torch
import os


class Converter:
    out_h, out_w = 256, 256

    def __init__(self, model_path, model_cfg, onnx_path="buffer/model.onnx", libtorch_path="buffer/model.pt",
                 onnx_sim_path="buffer/model_sim.onnx", device="cpu"):
        self.onnx_path = onnx_path
        self.libtorch_path = libtorch_path
        self.onnx_sim_path = onnx_sim_path

        posenet = PoseModel(device=device)
        posenet.build(model_cfg)
        posenet.load(model_path)
        self.model = posenet.model

        option_path = get_option_path(model_path)
        if os.path.exists(option_path):
            option = torch.load(option_path)
            self.out_h = option.output_height
            self.out_w = option.output_width

        if device != "cpu":
            self.dummy_input = torch.rand(2, 3, self.out_w, self.out_h).cuda()
        else:
            self.dummy_input = torch.rand(2, 3, self.out_w, self.out_h)

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

    def convert(self):
        with torch.no_grad():
            self.onnx_convert()
            self.libtorch_convert()


if __name__ == '__main__':
    model_path = "exp/test_kps/mpii_13/latest.pth"
    model_cfg = "exp/test_kps/mpii_13/model_cfg.json"

    if not model_path:
        model_cfg, _, _ = get_corresponding_cfg(model_path, check_exist=["model"])

    convert = Converter(model_path, model_cfg)
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

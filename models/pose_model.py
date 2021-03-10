import torch
from models.utils.benchmark import print_model_param_flops, print_model_param_nums, get_inference_time
from models.build import PoseNet
from .utils.utils import parse_cfg, parse_num_block


class PoseModel:
    def __init__(self, device="cuda:0"):
        self.is_freeze = False
        self.device = device

    def build(self, cfg):
        self.cfg = parse_cfg(cfg)
        self.backbone = self.cfg["backbone"]
        self.head = self.cfg["head_type"]
        self.kps = self.cfg["keypoints"]
        self.se_ratio = self.cfg["se_ratio"]
        self.block_nums = parse_num_block(self.cfg)
        self.first_conv = self.cfg["first_conv"]
        self.block_nums = parse_num_block(self.cfg)
        self.residual = self.cfg["residual"]
        self.head_channel = self.cfg["head_channel"]

        self.model = PoseNet(cfg, self.backbone, self.head)
        self.feature_layer_num, self.feature_layer_name = self.model.feature_layer_num, self.model.feature_layer_name
        if self.device != "cpu":
            self.model.cuda()

    def load(self, model_path, duc=False):
        if duc:
            out_dim = self.model.conv_out.out_channels
            self.model.conv_out = torch.nn.Conv2d(self.model.DIM, out_dim, kernel_size=3, stride=1, padding=1)
        self.model.load_state_dict(torch.load(model_path))

    def freeze(self, percent):
        if percent != 0:
            self.is_freeze = True
        feature_num = int(percent * self.feature_layer_num)
        for idx, (n, p) in enumerate(self.model.named_parameters()):
            if self.feature_layer_name in n and idx < feature_num:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def freeze_bn(self):
        for idx, (n, p) in enumerate(self.model.named_parameters()):
            if len(p.shape) == 1:
                p.requires_grad = False

    def init_with_opt(self, opt):
        self.build(opt.cfg)
        if opt.freeze_bn:
            self.freeze_bn()
        self.freeze(opt.freeze)
        if opt.loadModel:
            self.load(opt.loadModel)

    def benchmark(self, height=256, width=256):
        flops = print_model_param_flops(self.model)
        # print("FLOPs of current model is {}".format(flops))
        params = print_model_param_nums(self.model)
        # print("Parameters of current model is {}".format(params))
        inf_time = get_inference_time(self.model, height=height, width=width)
        # print("Inference time is {}".format(inf_time))
        # print("----------------------------------------------------------------------------------------------------")
        return flops, params, inf_time

    def get_updating_param(self):
        params_to_update, layers = [], 0
        for name, param in self.model.named_parameters():
            layers += 1
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update, layers
        #print("Training {} layers out of {}".format(len(params_to_update), layers))

    def model_transfer(self, device):
        if device != "cpu":
            self.model = torch.nn.DataParallel(self.model).cuda()
        else:
            self.model = torch.nn.DataParallel(self.model)


if __name__ == '__main__':
    PM = PoseModel(device="cpu")
    cfg = "./cfg/default/cfg_seresnet50.json"
    PM.build(cfg)
    net = PM.model
    y = net(torch.randn(1, 3, 320, 256))
    print(y.size())

import torch
from models.utils.benchmark import print_model_param_flops, print_model_param_nums, get_inference_time


class PoseModel:
    def __init__(self, device="cuda:0"):
        self.is_freeze = False
        self.device = device

    def build(self, backbone, cfg):
        self.backbone = backbone
        if backbone == "mobilenet":
            from models.mobilenet.MobilePose import createModel
            from config.model_cfg import mobile_opt as model_ls
            cfg = model_ls[cfg]
            self.feature_layer_num, self.feature_layer_name = 155, "features"
        elif backbone == "seresnet101":
            from models.seresnet.FastPose import createModel
            from config.model_cfg import seresnet_cfg as model_ls
            self.feature_layer_num, self.feature_layer_name = 327, "seresnet101"
        elif backbone == "efficientnet":
            from models.efficientnet.EfficientPose import createModel
            from config.model_cfg import efficientnet_cfg as model_ls
        elif backbone == "shufflenet":
            from models.shufflenet.ShufflePose import createModel
            from config.model_cfg import shufflenet_cfg as model_ls
            self.feature_layer_num, self.feature_layer_name = 167, "shuffle"
        elif backbone == "seresnet18":
            from models.seresnet18.FastPose import createModel
            from config.model_cfg import seresnet18_cfg as model_ls
            self.feature_layer_num, self.feature_layer_name = 75, "seresnet18"
        else:
            raise ValueError("Your model name is wrong")

        # self.model_cfg = cfg
        self.model = createModel(cfg)
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
        self.build(opt.backbone, opt.struct)
        if opt.freeze_bn:
            self.freeze_bn()
        self.freeze(opt.freeze)
        if opt.loadModel:
            self.load(opt.loadModel)

    def benchmark(self, height=256, width=256):
        flops = print_model_param_flops(self.model)
        print("FLOPs of current model is {}".format(flops))
        params = print_model_param_nums(self.model)
        print("Parameters of current model is {}".format(params))
        inf_time = get_inference_time(self.model, height=height, width=width)
        print("Inference time is {}".format(inf_time))
        print("----------------------------------------------------------------------------------------------------")
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

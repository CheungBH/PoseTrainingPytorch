from .utils.utils import parse_cfg
from torch import nn
from models.utils.default import *


class ModelBuilder:
    def __init__(self, cfg_file):
        self.cfg_file = cfg_file
        self.cfg = parse_cfg(cfg_file)
        self.backbone = self.cfg["backbone"]
        self.head = self.cfg["head_type"]
        self.model = self.build_model()

    def build_model(self):
        layers = []
        layers.append(self.build_backbone())
        layers.append(self.build_head())
        return nn.Sequential(*layers)

    def build(self):
        return self.model

    def build_backbone(self):
        if self.backbone == "mobilenet":
            from models.backbone.mobilenet import create
            self.feature_layer_num, self.feature_layer_name = 155, "features"
        elif self.backbone == "seresnet101":
            from models.backbone.seresnet101 import create
            self.feature_layer_num, self.feature_layer_name = 327, "seresnet101"
        elif self.backbone == "efficientnet":
            from models.efficientnet.EfficientPose import createModel as create
        elif self.backbone == "shufflenet":
            from models.backbone.shufflenet import shufflenet_v2_x1_0 as create
            self.feature_layer_num, self.feature_layer_name = 167, "shuffle"
        elif self.backbone == "seresnet18":
            from models.backbone.seresnet18 import create
            self.feature_layer_num, self.feature_layer_name = 75, "seresnet18"
        elif self.backbone == "seresnet50":
            from models.backbone.seresnet50 import create
            self.feature_layer_num, self.feature_layer_name = 75, "seresnet50"
        else:
            raise ValueError("Your model name is wrong")
        return create(self.cfg_file)

    def build_head(self):
        if self.head == "pixel_shuffle":
            from models.head.DUC import create
        return create(self.cfg_file)


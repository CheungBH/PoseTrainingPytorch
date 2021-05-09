from torch import nn


class PoseNet(nn.Module):
    def __init__(self, cfg_file, backbone, head):
        super(PoseNet, self).__init__()
        self.cfg = cfg_file
        self.backbone = self.build_backbone(backbone)
        self.head = self.build_head(head)
        a = 1

    def build_backbone(self, backbone):
        if backbone == "mobilenet":
            from models.backbone.mobilenet import create
            self.feature_layer_num, self.feature_layer_name = 155, "features"
        elif backbone == "seresnet101":
            from models.backbone.seresnet101 import create
            self.feature_layer_num, self.feature_layer_name = 327, "seresnet101"
        elif backbone == "efficientnet":
            from models.efficientnet.EfficientPose import createModel as create
        elif backbone == "shufflenet":
            from models.backbone.shufflenet import shufflenet_v2_x1_0 as create
            self.feature_layer_num, self.feature_layer_name = 167, "shuffle"
        elif backbone == "seresnet18":
            from models.backbone.seresnet18 import create
            self.feature_layer_num, self.feature_layer_name = 75, "seresnet18"
        elif backbone == "seresnet50":
            from models.backbone.seresnet50 import create
            self.feature_layer_num, self.feature_layer_name = 75, "seresnet50"
        else:
            raise ValueError("Your model name is wrong")
        return create(self.cfg)

    def build_head(self, head):
        if head == "pixel_shuffle":
            from models.head.DUC import create
        else:
            raise ValueError("Your model name is wrong")
        return create(self.cfg)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


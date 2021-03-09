from models.pose_model import PoseModel
from utils.prune_utils import sort_bn, obtain_bn_threshold, obtain_filters_mask, adjust_mask, get_residual_channel, \
    get_channel_dict
from models.utils.utils import write_cfg
import numpy as np
import torch

posenet = PoseModel(device="cpu")


class ShortcutPruner:
    def __init__(self, model_path, model_cfg, compact_model_path="", compact_model_cfg=""):
        self.model_path = model_path
        self.model_cfg = model_cfg

        posenet.build(model_cfg)
        self.model = posenet.model
        posenet.load(model_path)
        self.backbone = posenet.backbone
        self.kps = posenet.kps
        self.se_ratio = posenet.se_ratio

        if not compact_model_path or not compact_model_cfg:
            self.compact_model_path = "buffer/pruned_{}.pth".format(self.backbone)
            self.compact_model_cfg = "buffer/cfg_pruned_{}.json".format(self.backbone)
        else:
            self.compact_model_path = compact_model_path
            self.compact_model_cfg = compact_model_cfg

        if self.backbone == "seresnet18":
            from utils.prune_utils import obtain_prune_idx2 as obtain_prune
            self.init_weight = init_weights_from_loose_model
        elif self.backbone == "seresnet50" or self.backbone == "seresnet101":
            from utils.prune_utils import obtain_prune_idx_50 as obtain_prune
            self.init_weight = init_weights_from_loose_model50
        else:
            raise ValueError("{} is not supported for pruning! ".format(self.backbone))
        self.obtain_prune_idx = obtain_prune

    def run(self, threshold):


import os
import torch
from models.pose_model import PoseModel
from src.opt import opt
from utils.test_utils import check_option_file
from utils.prune_utils import obtain_prune_idx2, obtain_prune_idx_50, sort_bn, obtain_bn_threshold


class SparseDetector:
    backbone = "seresnet101"
    cfg = None
    opt.se_ratio = 16
    opt.kps = 17

    def __init__(self, model_path, device="cpu", thresh=(50, 99), step=1, method="ordinary", print_info=True):
        posenet = PoseModel()
        self.option_file = check_option_file(model_path)
        if os.path.exists(self.option_file):
            self.load_from_option()

        posenet.build(self.backbone, self.cfg)
        self.model = posenet.model
        posenet.load(model_path)
        if device != "cpu":
            self.model.cuda()

        self.method = method
        self.thresh_range = thresh
        self.step = step
        self.print = print_info

        self.sparse_dict = {}

    def load_from_option(self):
        self.option = torch.load(self.option_file)
        opt.kps = self.option.kps
        opt.se_ratio = self.option.se_ratio
        self.backbone = self.option.backbone
        self.cfg = self.option.struct
        self.DUC = self.option.DUC

    def detect(self):
        if self.backbone == "seresnet18":
            all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx2(self.model)
        elif self.backbone == "seresnet50":
            all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx_50(self.model)
        else:
            raise ValueError("Not a correct name")

        if self.method == "ordinary":
            prune_idx = normal_idx + head_idx
        elif self.method == "shortcut":
            prune_idx = all_bn_id
        else:
            raise ValueError("Wrong pruning method name! ")

        sorted_bn = sort_bn(self.model, prune_idx)
        percent_ls = range(self.thresh_range[0], self.thresh_range[1], self.step)
        for percent in percent_ls:
            threshold = obtain_bn_threshold(self.model, sorted_bn, percent/100)
            self.sparse_dict[percent] = threshold
            if self.print:
                print("{}---->{}".format(percent, threshold))

    def get_sparse_result(self):
        return self.sparse_dict


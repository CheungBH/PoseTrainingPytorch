import os
import torch
import torch.nn as nn
import numpy as np
from models.pose_model import PoseModel
from src.opt import opt
from utils.test_utils import check_option_file
from utils.prune_utils import obtain_prune_idx2, obtain_prune_idx_50


def obtain_prune_idx(path):
    lines = []
    with open(path, 'r') as f:
        file = f.readlines()
        for line in file:
            lines.append(line)

    idx = 0
    prune_idx = []
    for line in lines:
        if "):" in line:
            idx += 1
        if "BatchNorm2d" in line:
            # print(idx, line)
            prune_idx.append(idx)

    prune_idx = prune_idx[1:]  # 去除第一个bn1层
    return prune_idx


def sort_bn(model, prune_idx):
    size_list = [m.weight.data.shape[0] for idx, m in enumerate(model.modules()) if idx in prune_idx]
    # bn_layer = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    bn_prune_layers = [m for idx, m in enumerate(model.modules()) if idx in prune_idx]
    bn_weights = torch.zeros(sum(size_list))

    index = 0
    for module, size in zip(bn_prune_layers, size_list):
        bn_weights[index:(index + size)] = module.weight.data.abs().clone()
        index += size
    sorted_bn = torch.sort(bn_weights)[0]

    return sorted_bn


def obtain_bn_threshold(model, sorted_bn, percentage):
    thre_index = int(len(sorted_bn) * percentage)
    thre = sorted_bn[thre_index]
    return thre


def obtain_bn_mask(bn_module, thre, device="cpu"):
    if device != "cpu":
        thre = thre.cuda()
    mask = bn_module.weight.data.abs().ge(thre).float()

    return mask


def obtain_filters_mask(model, prune_idx, thre):
    pruned = 0
    bn_count = 0
    total = 0
    num_filters = []
    pruned_filters = []
    filters_mask = []
    pruned_maskers = []

    for idx, module in enumerate(model.modules()):
        if isinstance(module, nn.BatchNorm2d):
            if idx in prune_idx:
                mask = obtain_bn_mask(module, thre).cpu().numpy()
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                if remain == 0:  # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module.weight.data.abs().max()
                    mask = obtain_bn_mask(module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain
                    bn_count += 1
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')

                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())
            else:
                mask = np.ones(module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return pruned_filters, pruned_maskers


class SparseDetector:
    backbone = "seresnet101"
    cfg = None
    opt.se_ratio = 16
    opt.kps = 17

    def __init__(self, model_path, device="cpu", thresh=(50, 99), step=1, method="ordinary", print_info=True):
        posenet = PoseModel()
        posenet.build(self.backbone, self.cfg)
        self.model = posenet.model
        posenet.load(model_path)
        self.method = method
        self.thresh_range = thresh
        self.step = step
        self.print = print_info
        self.option_file = check_option_file(model_path)
        if os.path.exists(self.option_file):
            self.load_from_option()
        if device != "cpu":
            self.model.cuda()
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


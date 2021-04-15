import os
import torch
from models.pose_model import PoseModel
from src.opt import opt
from utils.test_utils import check_option_file
from utils.prune_utils import obtain_prune_idx2, obtain_prune_idx_50, sort_bn, obtain_bn_threshold, write_filters_mask
from utils.utils import get_superior_path


class SparseDetector:

    def __init__(self, model_path, model_cfg=None, device="cpu", thresh=(50, 99), step=1, method="shortcut", print_info=True,
                 mask_interval=5):
        self.option_file = check_option_file(model_path)

        posenet = PoseModel()
        posenet.build(model_cfg)
        self.model = posenet.model
        posenet.load(model_path)
        self.backbone = posenet.backbone
        if device != "cpu":
            self.model.cuda()

        self.method = method
        self.thresh_range = thresh
        self.step = step
        self.print = print_info
        self.mask_interval = mask_interval

        self.sparse_dict = {}
        self.mask_file = os.path.join(get_superior_path(model_path), "sparse_mask", "mask_{}-{}.txt".format(
            method, model_path.replace("\\", "/").split("/")[-1][:-4]))
        os.makedirs(os.path.join(get_superior_path(model_path), "sparse_mask"), exist_ok=True)

    def detect(self):
        if self.backbone == "seresnet18":
            all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx2(self.model)
        elif self.backbone == "seresnet50" or self.backbone == "seresnet101":
            all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx_50(self.model)
        else:
            raise ValueError("Not a correct name")

        if self.method == "ordinary":
            self.prune_idx = normal_idx + head_idx
        elif self.method == "shortcut":
            self.prune_idx = all_bn_id
        else:
            raise ValueError("Wrong pruning method name! ")

        sorted_bn = sort_bn(self.model, self.prune_idx)
        percent_ls = range(self.thresh_range[0], self.thresh_range[1], self.step)
        for percent in percent_ls:
            threshold = obtain_bn_threshold(self.model, sorted_bn, percent/100)
            self.sparse_dict[percent] = threshold.tolist()
            if percent % self.mask_interval == 0:
                self.write_mask(threshold, percent)
            if self.print:
                print("{}---->{}".format(percent, threshold))
        return self.sparse_dict

    def get_result_ls(self):
        result = [v for k, v in self.sparse_dict.items()]
        return result

    def write_mask(self, thresh, percent):
        with open(self.mask_file, "a+") as file:
            file.write("-----------------------Threshold {} at {}% pruned------------------------\n".format(thresh, percent))
            write_filters_mask(self.model, self.prune_idx, thresh, file)
            file.write("\n\n\n")


if __name__ == '__main__':
    model_path = "exp/test_structure/seres18_17kps/seres18_17kps_best_acc.pkl"
    model_cfg = "exp/test_structure/seres18_17kps/data_default.json"
    sd = SparseDetector(model_path, model_cfg)
    res = sd.detect()
    print(res)

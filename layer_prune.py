import torch
from models.pose_model import PoseModel
from utils.prune_utils import obtain_prune_idx_layer

posenet = PoseModel(device="cpu")


class LayerPruner:
    def __init__(self, model_path, model_cfg, compact_model_path="", compact_model_cfg=""):
        self.model_path = model_path
        self.model_cfg = model_cfg

        posenet.build(model_cfg)
        self.model = posenet.model
        posenet.load(model_path)
        self.backbone = posenet.backbone
        self.kps = posenet.kps
        self.se_ratio = posenet.se_ratio
        self.block_num = posenet.block_nums

        if not compact_model_path or not compact_model_cfg:
            self.compact_model_path = "buffer/layer_{}.pth".format(self.backbone)
            self.compact_model_cfg = "buffer/cfg_layer_{}.json".format(self.backbone)
        else:
            self.compact_model_path = compact_model_path
            self.compact_model_cfg = compact_model_cfg

    def run(self, prune_num):
        all_bn_id, other_idx, shortcut_idx, downsample_idx = obtain_prune_idx_layer(self.model)

        bn_mean = torch.zeros(len(shortcut_idx))
        for i, idx in enumerate(shortcut_idx):
            bn_mean[i] = list(self.model.named_modules())[idx][1].weight.data.abs().mean().clone()
        _, sorted_index_thre = torch.sort(bn_mean)
        prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:prune_num]]]
        prune_shortcuts = [int(x) for x in prune_shortcuts]
        a = 1


if __name__ == '__main__':
    model_path = "exp/test_structure/seres101/seres101_best_acc.pkl"
    model_cfg = 'exp/test_structure/seres101/cfg.json'
    LP = LayerPruner(model_path, model_cfg)
    LP.run(4)

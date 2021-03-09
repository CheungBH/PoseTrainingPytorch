import torch
from src.opt import opt
from models.pose_model import PoseModel
from utils.prune_utils import obtain_prune_idx_layer


backbone = "seresnet101"
cfg = None
height = 320
width = 256
opt.kps = 17
opt.se_ratio = 1


def layer_pruning(weight, compact_model_path, compact_model_cfg="cfg.txt", prune_num=4, device="cpu"):
    posenet = PoseModel(device=device)
    posenet.build(backbone, cfg)
    model = posenet.model
    posenet.load(weight)

    all_bn_id, other_idx, shortcut_idx, downsample_idx = obtain_prune_idx_layer(model)

    bn_mean = torch.zeros(len(shortcut_idx))
    for i, idx in enumerate(shortcut_idx):
        bn_mean[i] = list(model.named_modules())[idx][1].weight.data.abs().mean().clone()
    _, sorted_index_thre = torch.sort(bn_mean)
    prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:prune_num]]]
    prune_shortcuts = [int(x) for x in prune_shortcuts]
    a = 1


if __name__ == '__main__':
    layer_pruning("exp/seresnet101/sparse/sparse_40.pkl", "buffer/pruned_layer_{}.pth".format(opt.backbone),
                    "buffer/cfg_layer_{}.txt".format(opt.backbone))

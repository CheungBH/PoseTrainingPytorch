import torch
from models.pose_model import PoseModel
from utils.prune_utils import obtain_prune_idx_layer, obtain_channel_with_block_num
import numpy as np
from models.utils.utils import write_cfg
posenet = PoseModel(device="cpu")


def obtain_filters_mask(model, all_bn_idx, prune_layers):
    filters_mask = []
    for idx in all_bn_idx:
        bn_module = list(model.named_modules())[idx][1]
        mask = np.ones(bn_module.weight.data.shape[0], dtype='float32')
        filters_mask.append(mask.copy())
    CBLidx2mask = {idx: mask for idx, mask in zip(all_bn_idx, filters_mask)}
    for i in prune_layers:
        # for i in [idx, idx - 1]:
        bn_module = list(model.named_modules())[i][1]
        mask = np.zeros(bn_module.weight.data.shape[0], dtype='float32')
        CBLidx2mask[i] = mask.copy()
    return CBLidx2mask


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
        self.first_conv = posenet.first_conv
        self.residual = posenet.residual
        self.head_channel = posenet.head_channel
        self.head_type = posenet.head

        if not compact_model_path or not compact_model_cfg:
            self.compact_model_path = "buffer/layer_{}.pth".format(self.backbone)
            self.compact_model_cfg = "buffer/cfg_layer_{}.json".format(self.backbone)
        else:
            self.compact_model_path = compact_model_path
            self.compact_model_cfg = compact_model_cfg

    def obtain_block_idx(self, shortcut_idx, prune_shortcuts):
        shortcuts_location = []
        candidate_block_num = [i-1 for i in self.block_num]
        for shortcut in prune_shortcuts:
            shortcuts_location.append(self.get_layer_block(shortcut_idx.index(shortcut), candidate_block_num))
        return shortcuts_location

    @staticmethod
    def get_layer_block(idx, candidate_block_num):
        if idx < candidate_block_num[0]:
            return 0
        elif idx < candidate_block_num[1]:
            return 1
        elif idx < candidate_block_num[2]:
            return 2
        elif idx < candidate_block_num[3]:
            return 3
        else:
            raise IndexError("????????")

    def run(self, prune_num):
        all_bn_id, other_idx, shortcut_idx, downsample_idx = obtain_prune_idx_layer(self.model)

        bn_mean = torch.zeros(len(shortcut_idx))
        for i, idx in enumerate(shortcut_idx):
            bn_mean[i] = list(self.model.named_modules())[idx][1].weight.data.abs().mean().clone()
        _, sorted_index_thre = torch.sort(bn_mean)
        prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:prune_num]]]
        prune_shortcuts = [int(x) for x in prune_shortcuts]

        # prune_layers = []
        # for prune_shortcut in prune_shortcuts:
        #     target_idx = all_bn_id.index(prune_shortcut)
        #     for i in range(3):
        #         prune_layers.append(all_bn_id[target_idx-i])
        #
        # CBLidx2mask = obtain_filters_mask(self.model, all_bn_id, prune_layers)
        #
        pruned_locations = self.obtain_block_idx(shortcut_idx, prune_shortcuts)
        blocks = self.block_num
        for pruned_location in pruned_locations:
            blocks[pruned_location] -= 1

        m_cfg = {
            'backbone': self.backbone,
            'keypoints': self.kps,
            'se_ratio': self.se_ratio,
            "first_conv": self.first_conv,
            'residual': self.residual,
            'channels': obtain_channel_with_block_num(blocks),
            "head_type": self.head_type,
            "head_channel": self.head_channel
        }
        write_cfg(m_cfg, self.compact_model_cfg)
        posenet.build(self.compact_model_cfg)
        compact_model = posenet.model
        # self.init_weight(compact_model, self.model, CBLidx2mask, valid_filter, downsample_idx, head_idx)
        torch.save(compact_model.state_dict(), self.compact_model_path)


if __name__ == '__main__':
    model_path = "exp/test_structure/seres101/seres101_best_acc.pkl"
    model_cfg = 'exp/test_structure/seres101/cfg.json'
    LP = LayerPruner(model_path, model_cfg)
    LP.run(4)

import torch
from models.pose_model import PoseModel
from utils.prune_utils import obtain_prune_idx_layer, obtain_channel_with_block_num, \
    init_weights_from_loose_model_layer, print_mean, obtain_layer_filters_mask
from models.utils.utils import write_cfg

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
        for i in range(len(candidate_block_num)):
            if idx < sum(candidate_block_num[:i+1]):
                return i
        raise ValueError("Wrong index! ")

    def run(self, prune_num):
        all_bn_id, other_idx, shortcut_idx, downsample_idx = obtain_prune_idx_layer(self.model)

        bn_mean = torch.zeros(len(shortcut_idx))
        for i, idx in enumerate(shortcut_idx):
            bn_mean[i] = list(self.model.named_modules())[idx][1].weight.data.abs().mean().clone()
        _, sorted_index_thre = torch.sort(bn_mean)

        prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:prune_num]]]
        prune_shortcuts = [int(x) for x in prune_shortcuts]
        print_mean(bn_mean, shortcut_idx, prune_shortcuts)

        prune_layers = []
        for prune_shortcut in prune_shortcuts:
            target_idx = all_bn_id.index(prune_shortcut)
            for i in range(3):
                prune_layers.append(all_bn_id[target_idx-i])

        CBLidx2mask = obtain_layer_filters_mask(self.model, all_bn_id, prune_layers)

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
        # compact_all_bn = [idx for idx, mod in enumerate(list(compact_model.named_modules()))
        #                   if isinstance(mod[1], torch.nn.BatchNorm2d)]
        compact_all_bn_idx, compact_other_idx, compact_shortcut_idx, compact_downsample_idx = \
            obtain_prune_idx_layer(compact_model)
        init_weights_from_loose_model_layer(compact_model, self.model, CBLidx2mask, compact_all_bn_idx)
        torch.save(compact_model.state_dict(), self.compact_model_path)


if __name__ == '__main__':
    model_path = "exp/test_structure/seres50_17kps/seres50_17kps_best_acc.pkl"
    model_cfg = "exp/test_structure/seres50_17kps/data_default.json"
    LP = LayerPruner(model_path, model_cfg)
    LP.run(2)

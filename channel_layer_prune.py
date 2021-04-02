from models.pose_model import PoseModel
from utils.prune_utils import *
from models.utils.utils import write_cfg
import numpy as np
import torch

posenet = PoseModel(device="cpu")


class ChannelLayerPruner:
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
            self.compact_model_path = "buffer/all_{}.pth".format(self.backbone)
            self.compact_model_cfg = "buffer/cfg_all_{}.json".format(self.backbone)
        else:
            self.compact_model_path = compact_model_path
            self.compact_model_cfg = compact_model_cfg

        if self.backbone == "seresnet50" or self.backbone == "seresnet101":
            from utils.prune_utils import obtain_prune_idx_50 as obtain_prune
            self.init_weight_channel = init_weights_from_loose_model_shortcut50
            self.init_weight_layer = init_weights_from_loose_model_layer
        else:
            raise ValueError("{} is not supported for layer pruning! ".format(self.backbone))
        self.obtain_prune_idx = obtain_prune

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

    def run(self, threshold, layer_num):
        all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = self.obtain_prune_idx(self.model)
        prune_idx = all_bn_id
        sorted_bn = sort_bn(self.model, prune_idx)

        threshold = obtain_bn_threshold(self.model, sorted_bn, threshold / 100)
        pruned_filters, pruned_maskers = obtain_filters_mask(self.model, prune_idx, threshold)
        CBLidx2mask_channel = {idx - 1: mask.astype('float32') for idx, mask in zip(all_bn_id, pruned_maskers)}
        CBLidx2filter = {idx - 1: filter_num for idx, filter_num in zip(all_bn_id, pruned_filters)}

        final_layer_groups = [downsample_idx[-1] - 1] + [shortcut_idx[-1] - 1, shortcut_idx[-2] - 1]
        mask_groups = [
            [shortcut_idx[sum(self.block_num[:0]) + i] for i in range(self.block_num[0])] + [downsample_idx[0]],
            [shortcut_idx[sum(self.block_num[:1]) + i] for i in range(self.block_num[1])] + [downsample_idx[1]],
            [shortcut_idx[sum(self.block_num[:2]) + i] for i in range(self.block_num[2])] + [downsample_idx[2]],
            [shortcut_idx[sum(self.block_num[:3]) + i] for i in range(self.block_num[3])] + [downsample_idx[3]]]
        if self.backbone == "seresnet50" or self.backbone == "seresnet101":
            final_layer_groups.append(shortcut_idx[-3]-1)

        merge_mask(CBLidx2mask_channel, CBLidx2filter, mask_groups)
        adjust_final_mask(CBLidx2mask_channel, CBLidx2filter, self.model, final_layer_groups)
        for head in head_idx:
            adjust_mask(CBLidx2mask_channel, CBLidx2filter, self.model, head)

        valid_filter = {k: v for k, v in CBLidx2filter.items() if k + 1 in prune_idx}
        channel_str = ",".join(map(lambda x: str(x), valid_filter.values()))
        print(channel_str, file=open("buffer/cfg_all_{}.txt".format(self.backbone), "w"))
        m_channel_cfg = {
            'backbone': self.backbone,
            'keypoints': self.kps,
            'se_ratio': self.se_ratio,
            "first_conv": valid_filter[all_bn_id[0] - 1],
            'residual': get_residual_channel([filt for _, filt in valid_filter.items()], self.backbone),
            'channels': get_channel_dict([filt for _, filt in valid_filter.items()], self.backbone),
            "head_type": "pixel_shuffle",
            "head_channel": [CBLidx2filter[i - 1] for i in head_idx]
        }
        write_cfg(m_channel_cfg, self.compact_model_cfg)
        posenet.build(self.compact_model_cfg)
        compact_channel_model = posenet.model
        self.init_weight_channel(compact_channel_model, self.model, CBLidx2mask_channel, valid_filter, downsample_idx,
                                 head_idx)


        all_bn_id, other_idx, shortcut_idx, downsample_idx = obtain_prune_idx_layer(self.model)

        bn_mean = torch.zeros(len(shortcut_idx))
        for i, idx in enumerate(shortcut_idx):
            bn_mean[i] = list(self.model.named_modules())[idx][1].weight.data.abs().mean().clone()
        _, sorted_index_thre = torch.sort(bn_mean)

        prune_shortcuts = torch.tensor(shortcut_idx)[[sorted_index_thre[:layer_num]]]
        prune_shortcuts = [int(x) for x in prune_shortcuts]
        print_mean(bn_mean, shortcut_idx, prune_shortcuts)

        prune_layers = []
        for prune_shortcut in prune_shortcuts:
            target_idx = all_bn_id.index(prune_shortcut)
            for i in range(3):
                prune_layers.append(all_bn_id[target_idx - i])

        CBLidx2mask_layer = obtain_layer_filters_mask(compact_channel_model, all_bn_id, prune_layers)

        pruned_locations = self.obtain_block_idx(shortcut_idx, prune_shortcuts)
        blocks = self.block_num
        for pruned_location in pruned_locations:
            blocks[pruned_location] -= 1

        m_layer_cfg = {
            'backbone': self.backbone,
            'keypoints': self.kps,
            'se_ratio': self.se_ratio,
            "first_conv": m_channel_cfg["first_conv"],
            'residual': m_channel_cfg["residual"],
            'channels': obtain_all_prune_channels(sorted_index_thre[:layer_num].tolist(), m_channel_cfg["channels"]),
            "head_type": self.head_type,
            "head_channel": m_channel_cfg["head_channel"],
        }
        write_cfg(m_layer_cfg, self.compact_model_cfg)
        posenet.build(self.compact_model_cfg)
        compact_layer_model = posenet.model
        # compact_all_bn = [idx for idx, mod in enumerate(list(compact_model.named_modules()))
        #                   if isinstance(mod[1], torch.nn.BatchNorm2d)]
        compact_all_bn_idx, compact_other_idx, compact_shortcut_idx, compact_downsample_idx = \
            obtain_prune_idx_layer(compact_layer_model)
        init_weights_from_loose_model_layer(compact_layer_model, compact_channel_model, CBLidx2mask_layer, compact_all_bn_idx)

        torch.save(compact_layer_model.state_dict(), self.compact_model_path)


if __name__ == '__main__':
    model_path = "exp/test_structure/seres50_17kps/seres50_17kps_best_acc.pkl"
    model_cfg = "exp/test_structure/seres50_17kps/cfg.json"
    thresh = 80
    layer_num = 2
    SP = ChannelLayerPruner(model_path, model_cfg)
    SP.run(thresh, layer_num)

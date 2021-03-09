from models.pose_model import PoseModel
from utils.prune_utils import sort_bn, obtain_bn_threshold, obtain_filters_mask, adjust_mask, get_residual_channel, \
    get_channel_dict
from models.utils.utils import write_cfg
import numpy as np
import torch

posenet = PoseModel(device="cpu")


def init_weights_from_loose_model(compact_model, loose_model, CBLidx2mask, valid_filter, downsample_idx, head_idx):
    layer_nums = [k for k in CBLidx2mask.keys()]
    for idx, layer_num in enumerate(layer_nums):
        # if layer_num in valid_filter:
        out_channel_idx = np.argwhere(CBLidx2mask[layer_num])[:, 0].tolist()

        if idx == 0:
            in_channel_idx = [0, 1, 2]
        elif layer_num +1 in downsample_idx:
            last_conv_index = layer_nums[idx - 3]
            in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()
        elif layer_num +1 in head_idx:
            in_channel_idx = list(range(list(loose_model.named_modules())[layer_num][1].in_channels))
        else:
            last_conv_index = layer_nums[idx - 1]
            in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()

        compact_bn, loose_bn         = list(compact_model.modules())[layer_num+1], list(loose_model.modules())[layer_num+1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()
        #input mask is

        compact_conv, loose_conv = list(compact_model.modules())[layer_num], list(loose_model.modules())[layer_num]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    for layer_num, (layer_name, layer) in enumerate(list(loose_model.named_modules())):
        if "fc.0" in layer_name or "fc.2" in layer_name:
            compact_fc, loose_fc = list(compact_model.modules())[layer_num], list(loose_model.modules())[layer_num]
            compact_fc.weight.data = loose_fc.weight.data.clone()


def init_weights_from_loose_model50(compact_model, loose_model, CBLidx2mask, valid_filter, downsample_idx, head_idx):
    layer_nums = [k for k in CBLidx2mask.keys()]
    for idx, layer_num in enumerate(layer_nums):
        # if layer_num in valid_filter:
        out_channel_idx = np.argwhere(CBLidx2mask[layer_num])[:, 0].tolist()

        if idx == 0:
            in_channel_idx = [0, 1, 2]
        elif layer_num + 1 in downsample_idx:
            downsample_id = downsample_idx.index(layer_num+1)
            if downsample_id == 0:
                last_conv_index = layer_nums[0]
            else:
                last_conv_index = downsample_idx[downsample_id-1] - 1
            in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()
        elif layer_num + 1 in head_idx:
            break
            in_channel_idx = list(range(list(loose_model.named_modules())[layer_num][1].in_channels))
        else:
            last_conv_index = layer_nums[idx - 1]
            in_channel_idx = np.argwhere(CBLidx2mask[last_conv_index])[:, 0].tolist()

        compact_bn, loose_bn = list(compact_model.modules())[layer_num + 1], list(loose_model.modules())[layer_num + 1]
        compact_bn.weight.data = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data = loose_bn.running_var.data[out_channel_idx].clone()

        compact_conv, loose_conv = list(compact_model.modules())[layer_num], list(loose_model.modules())[layer_num]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()


class Pruner:
    def __init__(self, model_path, model_cfg, compact_model_path="", compact_model_cfg=""):
        self.model_path = model_path
        self.model_cfg = model_cfg

        posenet.build(model_cfg)
        self.model = posenet.model
        posenet.load(model_path)
        self.backbone = posenet.MB.backbone
        self.kps = posenet.MB.kps
        self.se_ratio = posenet.MB.se_ratio

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
        all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = self.obtain_prune_idx(self.model)
        prune_idx = normal_idx + head_idx
        sorted_bn = sort_bn(self.model, prune_idx)
        threshold = obtain_bn_threshold(self.model, sorted_bn, threshold/100)
        pruned_filters, pruned_maskers = obtain_filters_mask(self.model, prune_idx, threshold)

        CBLidx2mask = {idx - 1: mask.astype('float32') for idx, mask in zip(all_bn_id, pruned_maskers)}
        CBLidx2filter = {idx - 1: filter_num for idx, filter_num in zip(all_bn_id, pruned_filters)}

        for head in head_idx:
            adjust_mask(CBLidx2mask, CBLidx2filter, self.model, head)

        valid_filter = {k: v for k, v in CBLidx2filter.items() if k + 1 in prune_idx}
        channel_str = ",".join(map(lambda x: str(x), valid_filter.values()))
        print(channel_str, file=open("buffer/cfg_{}".format(self.backbone), "w"))
        m_cfg = {
            'backbone': self.backbone,
            'keypoints': self.kps,
            'se_ratio': self.se_ratio,
            "first_conv": CBLidx2filter[all_bn_id[0] - 1],
            'residual': get_residual_channel([filt for _, filt in valid_filter.items()], self.backbone),
            'channels': get_channel_dict([filt for _, filt in valid_filter.items()], self.backbone),
            "head_type": "pixel_shuffle",
            "head_channel": [CBLidx2filter[i - 1] for i in head_idx]
        }
        write_cfg(m_cfg, self.compact_model_cfg)

        posenet.build(self.compact_model_cfg)
        compact_model = posenet.model
        self.init_weight(compact_model, self.model_path, CBLidx2mask, valid_filter, downsample_idx, head_idx)
        torch.save(compact_model.state_dict(), self.compact_model_path)


if __name__ == '__main__':
    model_path = ""
    model_cfg = ""
    pruner = Pruner(model_path, model_cfg)
    pruner.run(80)

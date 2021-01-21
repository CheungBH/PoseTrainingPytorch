import os
import torch
import torch.nn as nn
import numpy as np
# from config.config import device
from models.seresnet.FastPose import createModel
from src.opt import opt


def obtain_prune_idx(path):
    lines = []
    with open(path, 'r') as f:
        file = f.readlines()
        for line in file:
            lines.append(line)

    idx = 0
    prune_idx = []
    bn3_id = []
    for line in lines:
        if "):" in line:
            idx += 1
        if "BatchNorm2d" in line and "bn3" not in line:
            # print(idx, line)
            prune_idx.append(idx)
        # if "(bn3)" in line:
        #     bn3_id.append(idx)


    prune_idx = prune_idx[1:]  # 去除第一个bn1层
    return prune_idx,bn3_id


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
                pruned_filters.append(remain)

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return pruned_filters[1:], pruned_maskers

def init_weights_from_loose_model(compact_model, loose_model, CBL_idx, Conv_idx, CBLidx2mask):

    for i,idx in enumerate(CBL_idx):
        # compact_CBL = list(compact_model.modules())[idx]
        # loose_CBL = list(loose_model.modules())[idx]
        out_channel_idx = np.argwhere(CBLidx2mask[idx])[:, 0].tolist()

        compact_bn, loose_bn         = list(compact_model.modules())[idx+1], list(loose_model.modules())[idx+1]
        compact_bn.weight.data       = loose_bn.weight.data[out_channel_idx].clone()
        compact_bn.bias.data         = loose_bn.bias.data[out_channel_idx].clone()
        compact_bn.running_mean.data = loose_bn.running_mean.data[out_channel_idx].clone()
        compact_bn.running_var.data  = loose_bn.running_var.data[out_channel_idx].clone()
        #input mask is

        # input_mask = get_input_mask(loose_model.module_defs, idx, CBLidx2mask)
        # if i==0:
        #     in_channel_idx = list(range(64))
        # else:
        in_channel_idx = np.argwhere(CBLidx2mask[CBL_idx[i-1]])[:, 0].tolist()
        compact_conv, loose_conv = list(compact_model.modules())[idx], list(loose_model.modules())[idx]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()
        print(idx)

def pruning(weight, thresh=80, device="cpu"):
    if opt.backbone == "mobilenet":
        from models.mobilenet.MobilePose import createModel
        from config.model_cfg import mobile_opt as model_ls
    elif opt.backbone == "seresnet101":
        from models.seresnet.FastPose import createModel
        from config.model_cfg import seresnet_cfg as model_ls
    elif opt.backbone == "efficientnet":
        from models.efficientnet.EfficientPose import createModel
        from config.model_cfg import efficientnet_cfg as model_ls
    elif opt.backbone == "shufflenet":
        from models.shufflenet.ShufflePose import createModel
        from config.model_cfg import shufflenet_cfg as model_ls
    else:
        raise ValueError("Your model name is wrong")
    model_cfg = model_ls[opt.struct]
    # opt.loadModel = weight

    model = createModel(cfg=model_cfg)
    model.load_state_dict(torch.load(weight))
    if device == "cpu":
        model.cpu()
    else:
        model.cuda()

    tmp = "./model.txt"
    print(model, file=open(tmp, 'w'))
    prune_idx, bn3_id = obtain_prune_idx(tmp)
    Conv_idx = [conv-1 for conv in prune_idx]
    sorted_bn = sort_bn(model, prune_idx)

    threshold = obtain_bn_threshold(model, sorted_bn, thresh/100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)
    CBLidx2mask = {idx: mask.astype('float32') for idx, mask in zip(Conv_idx, pruned_maskers)}
    print(pruned_filters, file=open("ceiling.txt", "w"))
    new_model = createModel(cfg="ceiling.txt").cpu()

    init_weights_from_loose_model(new_model, model, Conv_idx, Conv_idx, CBLidx2mask)

    print()


if __name__ == '__main__':
    pruning("/media/hkuit164/MB155_1/sparsed_demo/origin_5E-4-acc/origin_5E-4_best_acc.pkl")
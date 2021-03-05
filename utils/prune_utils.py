import torch.nn as nn
import numpy as np
import torch


def obtain_prune_idx_50(model):
    all_bn_id, normal_idx, head_idx, shortcut_idx, downsample_idx = [], [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "seresnet50" in layer[0] or "preact" in layer[0]:
                if "downsample" in layer[0]:
                    downsample_idx.append(i)
                elif "bn1" in layer[0] or "bn2" in layer[0] and i > 5:
                    normal_idx.append(i)
                elif "bn3" in layer[0]:
                    shortcut_idx.append(i)
                else:
                    print("???????")
            else:
                head_idx.append(i)
    return all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx


def obtain_prune_idx2(model):
    all_bn_id, normal_idx, head_idx, shortcut_idx, downsample_idx = [], [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "seresnet18" in layer[0]:
                if i < 5:
                    shortcut_idx.append(i)
                elif "downsample" in layer[0]:
                    downsample_idx.append(i)
                elif "bn1" in layer[0] and i > 5:
                    normal_idx.append(i)
                elif "bn3" in layer[0]:
                    shortcut_idx.append(i)
                else:
                    print("???????")
            else:
                head_idx.append(i)
    return all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx


def obtain_prune_idx_layer(model):
    all_bn_id, other_idx, shortcut_idx, downsample_idx = [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "bn3" in layer[0]:
                if ".0." in layer[0]:
                    downsample_idx.append(i)
                else:
                    shortcut_idx.append(i)
            else:
                other_idx.append(i)
    return all_bn_id, other_idx, shortcut_idx, downsample_idx

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


def write_filters_mask(model, prune_idx, thre, file):
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

                if remain == 0:  # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module.weight.data.abs().max()
                    mask = obtain_bn_mask(module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    # pruned = pruned + mask.shape[0] - remain
                    bn_count += 1
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}', file=file)

                pruned = pruned + mask.shape[0] - remain
                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())
                total += mask.shape[0]
                num_filters.append(remain)
                filters_mask.append(mask.copy())
            else:

                mask = np.ones(module.weight.data.shape)
                remain = mask.shape[0]
                pruned_filters.append(remain)
                pruned_maskers.append(mask.copy())

    prune_ratio = pruned / total
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}', file=file)

    return pruned_filters, pruned_maskers


def get_residual_channel(channel_ls, backbone):
    if backbone == "seresnet18":
        if len(channel_ls) < 10:
            return [64, 128, 256, 512]
        else:
            return [channel_ls[4], channel_ls[9], channel_ls[14], channel_ls[19]]
    elif backbone == "seresnet50":
        if len(channel_ls) < 40:
            return [256, 512, 1024, 2048]
        else:
            return [channel_ls[3], channel_ls[13], channel_ls[26], channel_ls[45]]
    elif backbone == "seresnet101":
        if len(channel_ls) < 100:
            return [256, 512, 1024, 2048]
        else:
            return [channel_ls[3], channel_ls[13], channel_ls[26], channel_ls[103]]


def get_channel_dict(channel_ls, backbone):
    if backbone == "seresnet18":
        if len(channel_ls) < 10:
            return {1: [[64]*2], 2: [[128]*2], 3:[[256]*2], 4: [[512]*2]}
        else:
            return {1: [[channel_ls[1]], [channel_ls[2]]],
                    2: [[channel_ls[3]], [channel_ls[4]]],
                    3: [[channel_ls[5]], [channel_ls[6]]],
                    4: [[channel_ls[7]], [channel_ls[8]]]}
    elif backbone == "seresnet50":
        cl = channel_ls[:1]
        if len(channel_ls) < 40:
            return {1: [[cl[2*i], cl[2*i+1]] for i in range(3)],
                    2: [[cl[6+2*i], cl[6+2*i+1]] for i in range(4)],
                    3: [[cl[14+2*i], cl[14+2*i+1]] for i in range(6)],
                    4: [[cl[26+2*i], cl[26+2*i+1]] for i in range(3)]}
        else:
            cl = channel_ls
            return {1: [[cl[1], cl[2]], [cl[5], cl[6]],[cl[8], cl[9]]],
                    2: [[cl[11], cl[12]], [cl[15], cl[16]], [cl[18], cl[19]], [cl[21], cl[22]]],
                    3: [[cl[24], cl[25]], [cl[28], cl[29]], [cl[31], cl[32]], [cl[34], cl[35]], [cl[37], cl[38]], [cl[40], cl[41]]],
                    4: [[cl[43], cl[44]], [cl[47], cl[48]], [cl[50], cl[51]]]
            }
    elif backbone == "seresnet101":
        if len(channel_ls) < 100:
            cl = channel_ls[1:]
            return {1: [[cl[2*i], cl[2*i+1]] for i in range(3)],
                    2: [[cl[6+2*i], cl[6+2*i+1]] for i in range(4)],
                    3: [[cl[14+2*i], cl[14+2*i+1]] for i in range(23)],
                    4: [[cl[60+2*i], cl[60+2*i+1]] for i in range(3)]}
        else:
            cl = channel_ls
            lay3_idx = [24, 25] + [idx for idx in range(106)[28:93] if idx % 3 != 0]
            return {1: [[cl[1], cl[2]], [cl[5], cl[6]],[cl[8], cl[9]]],
                    2: [[cl[11], cl[12]], [cl[15], cl[16]], [cl[18], cl[19]], [cl[21], cl[22]]],
                    3: [[cl[lay3_idx[2*i]], cl[lay3_idx[2*i+1]]] for i in range(int(len(lay3_idx)/2))],
                    4: [[cl[94], cl[95]], [cl[98], cl[99]], [cl[101], cl[102]]]
            }
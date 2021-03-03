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

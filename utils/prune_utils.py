import torch.nn as nn
import numpy as np
import torch


def obtain_prune_idx_50(model):
    all_bn_id, normal_idx, head_idx, shortcut_idx, downsample_idx = [], [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "backbone" in layer[0]:
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
            if "backbone" in layer[0]:
                if i < 5 or "downsample" in layer[0]:
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
    total = 0
    num_filters = []
    pruned_filters = []
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
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')

                pruned = pruned + mask.shape[0] - remain
                total += mask.shape[0]

            else:
                mask = np.ones(module.weight.data.shape)
                remain = mask.shape[0]

            pruned_filters.append(remain)
            pruned_maskers.append(mask.copy())

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


def adjust_mask(CBLidx2mask, CBLidx2filter, model, head_idx):
    CNN_weight = list(model.named_modules())[head_idx][1].weight.data.abs().clone()
    CNN_mask = CBLidx2mask[head_idx-1]
    num = CBLidx2filter[head_idx-1]
    total_num = len(CNN_weight.tolist())
    for idx in range(total_num):
        if CNN_mask[idx] == 1:
            CNN_weight[idx] = 0
    remaining_idx = 4 - num % 4
    _, sorted_idx = CNN_weight.sort(descending=True)
    padding_idx = sorted_idx[:remaining_idx]
    CBLidx2filter[head_idx-1] = num + remaining_idx
    for idx in padding_idx:
        CBLidx2mask[head_idx-1][idx] = 1


def adjust_final_mask(CBLidx2mask, CBLidx2filter, model, final_layer_groups):
    # if backbone == "seresnet18":
    #     #     final_layer_group, final_conv_idx = [77, 87, 93], 93
    #     # elif backbone == "seresnet50":
    #     #     final_layer_group, final_conv_idx = [136, 146, 153, 160], 160
    #     # elif backbone == "seresnet101":
    #     #     final_layer_group, final_conv_idx = [255, 265, 272, 279], 279

    final_conv_idx = final_layer_groups[-1]

    final_CNN_weight = list(model.named_modules())[final_conv_idx+1][1].weight.data.abs().clone()
    final_CNN_mask = CBLidx2mask[final_conv_idx]
    num = CBLidx2filter[final_conv_idx]
    total_num = len(final_CNN_weight.tolist())
    for idx in range(total_num):
        if final_CNN_mask[idx] == 1:
            final_CNN_weight[idx] = 0

    remaining_idx = 4 - num % 4
    _, sorted_idx = final_CNN_weight.sort(descending=True)
    padding_idx = sorted_idx[:remaining_idx]
    for layer in final_layer_groups:
        CBLidx2filter[layer] = num + remaining_idx
        for idx in padding_idx:
            CBLidx2mask[layer][idx] = 1


def merge_mask(CBLidx2mask, CBLidx2filter, mask_groups):
    # if backbone == "seresnet50":
    #     mask_groups = [[13, 23, 30, 37],
    #                    [45, 55, 62, 69, 76],
    #                    [84, 94, 101, 108, 115, 122, 129],
    #                    [137, 147, 154, 161]]
    # elif backbone == "seresnet18":
    #     mask_groups = [[2,11,24], [41,31,47],[64,54,70],[77,87,93]]
    # elif backbone == "seresnet101":
    #     mask_groups = [[13, 23, 30, 37],
    #                    [45, 55, 62, 69, 76],
    #                    [84, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 178, 185, 192, 199, 206, 213,
    #                     220, 227, 234, 241, 248],
    #                    [256, 266, 273, 280]]

    for layers in mask_groups:
        Merge_masks = []
        for layer in layers:
            Merge_masks.append(torch.Tensor(CBLidx2mask[layer-1]).unsqueeze(0))

        Merge_masks = torch.cat(Merge_masks, 0)
        merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()

        filter_num = int(torch.sum(merge_mask).item())
        merge_mask = np.array(merge_mask)

        for layer in layers:
            CBLidx2mask[layer-1] = merge_mask
            CBLidx2filter[layer-1] = filter_num

    return CBLidx2mask, CBLidx2filter


# def merge_mask(CBLidx2mask, CBLidx2filter, backbone):
#     if backbone == "seresnet18":
#         mask_groups = [[2,11,24], [41,31,47],[64,54,70],[77,87,93]]
#
#     for layer1, layer2, layer3 in mask_groups:
#         Merge_masks = []
#         Merge_masks.append(torch.Tensor(CBLidx2mask[layer1]).unsqueeze(0))
#         Merge_masks.append(torch.Tensor(CBLidx2mask[layer2]).unsqueeze(0))
#         Merge_masks.append(torch.Tensor(CBLidx2mask[layer3]).unsqueeze(0))
#         Merge_masks = torch.cat(Merge_masks, 0)
#         merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()
#
#         filter_num = int(torch.sum(merge_mask).item())
#         merge_mask = np.array(merge_mask)
#         CBLidx2mask[layer1] = merge_mask
#         CBLidx2mask[layer2] = merge_mask
#         CBLidx2mask[layer3] = merge_mask
#
#         CBLidx2filter[layer1] = filter_num
#         CBLidx2filter[layer2] = filter_num
#         CBLidx2filter[layer3] = filter_num
#     return CBLidx2mask, CBLidx2filter


def get_residual_channel(channel_ls, backbone):
    if backbone == "seresnet18":
        if len(channel_ls) < 12:
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
        if len(channel_ls) < 12:
            return {1: [[channel_ls[0]], [channel_ls[1]]],
                    2: [[channel_ls[2]], [channel_ls[3]]],
                    3: [[channel_ls[4]], [channel_ls[5]]],
                    4: [[channel_ls[6]], [channel_ls[7]]]}
        else:
            return {1: [[channel_ls[1]], [channel_ls[3]]],
                    2: [[channel_ls[5]], [channel_ls[8]]],
                    3: [[channel_ls[10]], [channel_ls[13]]],
                    4: [[channel_ls[15]], [channel_ls[18]]]}
    elif backbone == "seresnet50":
        cl = channel_ls[1:]
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

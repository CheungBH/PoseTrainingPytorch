import torch
import torch.nn as nn
import numpy as np
# from config.config import device
from utils.prune_utils import obtain_prune_idx2, obtain_prune_idx_50, get_residual_channel, get_channel_dict
from config.opt import opt
from models.utils.utils import write_cfg


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

    prune_idx = prune_idx  # 去除第一个bn1层
    return prune_idx, bn3_id


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


def adjust_final_mask(CBLidx2mask, CBLidx2filter, model):
    if opt.backbone == "seresnet18":
        final_layer_group, final_conv_idx = [77, 87, 93], 93
    elif opt.backbone == "seresnet50":
        final_layer_group, final_conv_idx = [136, 146, 153, 160], 160
    elif opt.backbone == "seresnet101":
        final_layer_group, final_conv_idx = [255, 265, 272, 279], 279

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
    for layer in final_layer_group:
        CBLidx2filter[layer] = num + remaining_idx
        for idx in padding_idx:
            CBLidx2mask[layer][idx] = 1


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

                if remain == 0:  # 保证至少有一个channel
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = module.weight.data.abs().max()
                    mask = obtain_bn_mask(module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    # pruned = pruned + mask.shape[0] - remain
                    bn_count += 1
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')

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
    print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')

    return pruned_filters, pruned_maskers


def init_weights_from_loose_model_shortcut(compact_model, loose_model, CBLidx2mask, valid_filter, downsample_idx, head_idx):
    layer_nums = [k for k in CBLidx2mask.keys()]
    for idx, layer_num in enumerate(layer_nums):
        # if layer_num in valid_filter:
        out_channel_idx = np.argwhere(CBLidx2mask[layer_num])[:, 0].tolist()

        if idx == 0:
            in_channel_idx = [0, 1, 2]
        elif layer_num + 1 in downsample_idx:
            last_conv_index = layer_nums[idx - 3]
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
        # input mask is

        compact_conv, loose_conv = list(compact_model.modules())[layer_num], list(loose_model.modules())[layer_num]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()

    # for layer_num, (layer_name, layer) in enumerate(list(loose_model.named_modules())):
        #     if "fc.0" in layer_name or "fc.2" in layer_name:
        #     compact_fc, loose_fc = list(compact_model.modules())[layer_num], list(loose_model.modules())[layer_num]
    #     compact_fc.weight.data = loose_fc.weight.data.clone()


def init_weights_from_loose_model_shortcut50(compact_model, loose_model, CBLidx2mask, valid_filter, downsample_idx, head_idx):
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
        # input mask is

        compact_conv, loose_conv = list(compact_model.modules())[layer_num], list(loose_model.modules())[layer_num]
        tmp = loose_conv.weight.data[:, in_channel_idx, :, :].clone()
        compact_conv.weight.data = tmp[out_channel_idx, :, :, :].clone()


def merge_mask50(CBLidx2mask, CBLidx2filter):
    if opt.backbone == "seresnet50":
        mask_groups = [[13, 23, 30, 37],
                       [45, 55, 62, 69, 76],
                       [84, 94, 101, 108, 115, 122, 129],
                       [137, 147, 154, 161]]
    elif opt.backbone == "seresnet18":
        mask_groups = [[2,11,24], [41,31,47],[64,54,70],[77,87,93]]
    elif opt.backbone == "seresnet101":
        mask_groups = [[13, 23, 30, 37],
                       [45, 55, 62, 69, 76],
                       [84, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 178, 185, 192, 199, 206, 213,
                        220, 227, 234, 241, 248],
                       [256, 266, 273, 280]]

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


def merge_mask(CBLidx2mask, CBLidx2filter):
    if opt.backbone == "seresnet18":
        mask_groups = [[2,11,24], [41,31,47],[64,54,70],[77,87,93]]

    for layer1, layer2, layer3 in mask_groups:
        Merge_masks = []
        Merge_masks.append(torch.Tensor(CBLidx2mask[layer1]).unsqueeze(0))
        Merge_masks.append(torch.Tensor(CBLidx2mask[layer2]).unsqueeze(0))
        Merge_masks.append(torch.Tensor(CBLidx2mask[layer3]).unsqueeze(0))
        Merge_masks = torch.cat(Merge_masks, 0)
        merge_mask = (torch.sum(Merge_masks, dim=0) > 0).float()

        filter_num = int(torch.sum(merge_mask).item())
        merge_mask = np.array(merge_mask)
        CBLidx2mask[layer1] = merge_mask
        CBLidx2mask[layer2] = merge_mask
        CBLidx2mask[layer3] = merge_mask

        CBLidx2filter[layer1] = filter_num
        CBLidx2filter[layer2] = filter_num
        CBLidx2filter[layer3] = filter_num
    return CBLidx2mask, CBLidx2filter


def pruning(weight, compact_model_path, compact_model_cfg="cfg.txt", thresh=80, device="cpu"):
    if opt.backbone == "mobilenet":
        from models.mobilenet.MobilePose import createModel
        from config.model_cfg import mobile_opt as model_ls
    elif opt.backbone == "seresnet101":
        from models.seresnet101.FastPose import createModel
        from config.model_cfg import seresnet_cfg as model_ls
    elif opt.backbone == "seresnet18":
        from models.seresnet18.FastPose import createModel
        from config.model_cfg import seresnet_cfg as model_ls
    elif opt.backbone == "efficientnet":
        from models.efficientnet.EfficientPose import createModel
        from config.model_cfg import efficientnet_cfg as model_ls
    elif opt.backbone == "shufflenet":
        from models.shufflenet.ShufflePose import createModel
        from config.model_cfg import shufflenet_cfg as model_ls
    elif opt.backbone == "seresnet50":
        from models.seresnet50.FastPose import createModel
        from config.model_cfg import seresnet50_cfg as model_ls
    else:
        raise ValueError("Your model name is wrong")
    # opt.loadModel = weight

    try:
        model_cfg = model_ls[opt.struct]
        model = createModel(cfg=model_cfg)
    except:
        model = createModel(cfg=opt.struct)

    model.load_state_dict(torch.load(weight))
    if device == "cpu":
        model.cpu()
    else:
        model.cuda()
    # torch_out = torch.onnx.export(model, torch.rand(1, 3, 224, 224), "onnx_pose.onnx", verbose=False,)

    tmp = "./buffer/model.txt"
    print(model, file=open(tmp, 'w'))
    if opt.backbone == "seresnet18":
        all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx2(model)
    elif opt.backbone == "seresnet50" or opt.backbone == "seresnet101":
        all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx_50(model)
    else:
        raise ValueError("Not a correct name")


    prune_idx = all_bn_id
    sorted_bn = sort_bn(model, prune_idx)

    threshold = obtain_bn_threshold(model, sorted_bn, thresh / 100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)
    CBLidx2mask = {idx - 1: mask.astype('float32') for idx, mask in zip(all_bn_id, pruned_maskers)}
    CBLidx2filter = {idx - 1: filter_num for idx, filter_num in zip(all_bn_id, pruned_filters)}
    if opt.backbone == "seresnet18":
        merge_mask(CBLidx2mask, CBLidx2filter)
    elif opt.backbone == "seresnet50" or opt.backbone == "seresnet101":
        merge_mask50(CBLidx2mask, CBLidx2filter)

    adjust_final_mask(CBLidx2mask, CBLidx2filter, model)
    for head in head_idx:
        adjust_mask(CBLidx2mask, CBLidx2filter, model, head)

    valid_filter = {k: v for k, v in CBLidx2filter.items() if k + 1 in prune_idx}
    channel_str = ",".join(map(lambda x: str(x), valid_filter.values()))
    print(channel_str, file=open(compact_model_cfg, "w"))
    m_cfg = {
        'backbone': opt.backbone,
        'keypoints': opt.kps,
        'se_ratio': opt.se_ratio,
        "first_conv": valid_filter[all_bn_id[0]-1],
        'residual': get_residual_channel([filt for _, filt in valid_filter.items()], opt.backbone),
        'channels': get_channel_dict([filt for _, filt in valid_filter.items()], opt.backbone),
        "head_type": "pixel_shuffle",
        "head_channel": [CBLidx2filter[i-1] for i in head_idx]
    }
    write_cfg(m_cfg, "buffer/cfg_shortcut_{}.json".format(opt.backbone))

    compact_model = createModel(cfg=compact_model_cfg).cpu()

    if opt.backbone == "seresnet18":
        init_weights_from_loose_model_shortcut(compact_model, model, CBLidx2mask, valid_filter, downsample_idx, head_idx)
    elif opt.backbone == "seresnet50" or opt.backbone == "seresnet101":
        init_weights_from_loose_model_shortcut50(compact_model, model, CBLidx2mask, valid_filter, downsample_idx, head_idx)
    torch.save(compact_model.state_dict(), compact_model_path)


if __name__ == '__main__':
    opt.backbone = "seresnet101"
    opt.se_ratio = 1
    opt.kps = 17
    pruning("exp/seresnet101/sparse/sparse_40.pkl", "buffer/pruned_shortcut_{}.pth".format(opt.backbone),
            "buffer/cfg_shortcut_{}.txt".format(opt.backbone))

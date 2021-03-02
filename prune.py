import os
import torch
import torch.nn as nn
import numpy as np
# from config.config import device
from utils.prune_utils import obtain_prune_idx2, obtain_prune_idx_50
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


    prune_idx = prune_idx  # 去除第一个bn1层
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


def pruning(weight, compact_model_path, compact_model_cfg="cfg.txt", thresh=80, device="cpu"):
    if opt.backbone == "mobilenet":
        from models.mobilenet.MobilePose import createModel
        from config.model_cfg import mobile_opt as model_ls
    elif opt.backbone == "seresnet101":
        from models.seresnet.FastPose import createModel
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
    model_cfg = model_ls[opt.struct]
    # opt.loadModel = weight

    model = createModel(cfg=model_cfg)
    model.load_state_dict(torch.load(weight))
    if device == "cpu":
        model.cpu()
    else:
        model.cuda()
    # torch_out = torch.onnx.export(model, torch.rand(1, 3, 224, 224), "onnx_pose.onnx", verbose=False,)

    tmp = "./model.txt"
    print(model, file=open(tmp, 'w'))
    if opt.backbone == "seresnet18":
        all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx2(model)
    elif opt.backbone == "seresnet50":
        all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx = obtain_prune_idx_50(model)
    else:
        raise ValueError("Not a correct name")
    prune_idx = normal_idx + head_idx
    sorted_bn = sort_bn(model, prune_idx)

    threshold = obtain_bn_threshold(model, sorted_bn, thresh/100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)
    CBLidx2mask = {idx-1: mask.astype('float32') for idx, mask in zip(all_bn_id, pruned_maskers)}
    CBLidx2filter = {idx-1: filter_num for idx, filter_num in zip(all_bn_id, pruned_filters)}

    valid_filter = {k: v for k, v in CBLidx2filter.items() if k+1 in prune_idx}
    channel_str = ",".join(map(lambda x: str(x), valid_filter.values()))
    print(channel_str, file=open(compact_model_cfg, "w"))
    compact_model = createModel(cfg=compact_model_cfg).cpu()
    print(compact_model, file=open("pruned.txt", 'w'))

    if opt.backbone == "seresnet18":
        init_weights_from_loose_model(compact_model, model, CBLidx2mask, valid_filter, downsample_idx, head_idx)
    elif opt.backbone == "seresnet50":
        init_weights_from_loose_model50(compact_model, model, CBLidx2mask, valid_filter, downsample_idx, head_idx)
    torch.save(compact_model.state_dict(), compact_model_path)


if __name__ == '__main__':
    opt.backbone = "seresnet50"
    opt.se_ratio = 16
    opt.kps = 17
    pruning("exp/resnet_test/aic_origin/aic_origin_best_acc.pkl", "pruned_{}.pth".format(opt.backbone), "cfg_{}.txt".format(opt.backbone))

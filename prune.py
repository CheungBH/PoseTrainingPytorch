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
    for line in lines:
        if "):" in line:
            idx += 1
        if "BatchNorm2d" in line:
            # print(idx, line)
            prune_idx.append(idx)

    prune_idx = prune_idx[1:]  # 去除第一个bn1层
    return prune_idx


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
    prune_idx = obtain_prune_idx(tmp)
    sorted_bn = sort_bn(model, prune_idx)

    threshold = obtain_bn_threshold(model, sorted_bn, thresh/100)
    pruned_filters, pruned_maskers = obtain_filters_mask(model, prune_idx, threshold)

    print(pruned_filters, file=open("ceiling.txt", "w"))
    new_model = createModel(cfg="ceiling.txt").cpu()


if __name__ == '__main__':
    pruning("test_weight/ceiling_0911/to17kps_s5E-7-acc/to17kps_s5E-7_best_acc.pkl")
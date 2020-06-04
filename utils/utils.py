import torch
from src.opt import opt


def gather_bn_weights(module_list, prune_idx):

    size_list = [module_list[idx][1].weight.data.shape[0] for idx in prune_idx]

    bn_weights = torch.zeros(sum(size_list))
    index = 0
    for idx, size in zip(prune_idx, size_list):
        bn_weights[index:(index + size)] = module_list[idx][1].weight.data.abs().clone()
        index += size

    return bn_weights


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1]


def adjust_lr(optimizer, epoch, lr_dict, nEpoch):
    curr_ratio = epoch/nEpoch
    bound = list(lr_dict.keys())
    if curr_ratio > bound[0] and curr_ratio <= bound[1]:
        lr = opt.LR * lr_dict[bound[0]]
    elif curr_ratio > bound[1]:
        lr = opt.LR * lr_dict[bound[1]]
    else:
        lr = opt.LR

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return optimizer, lr

import torch
from src.opt import opt
from config import config


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
    return string[:-1] + "\n"


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


def lr_decay(optimizer, lr):
    lr = lr * 0.1
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return optimizer, lr


def warm_up_lr(optimizer, epoch):
    bound = sorted(list(config.warm_up.keys()))
    if epoch < bound[0]:
        lr = opt.LR * config.warm_up[bound[0]]
    elif epoch < bound[1]:
        lr = opt.LR * config.warm_up[bound[1]]
    else:
        lr = opt.LR

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return optimizer, lr


def get_sparse_value():
    if opt.epoch > opt.nEpochs * config.sparse_decay_time:
        return opt.sparse_s * opt.sparse_decay
    return opt.sparse_s


def write_csv_title():
    title = ["epoch", "lr", "train_loss", "train_acc"]
    title += csv_body_part("train")
    title += [" ", "val_loss", "val_acc"]
    title += csv_body_part("val")
    return title


def csv_body_part(phase):
    ls = []
    for item in config.body_part_name:
        ls.append(phase+"_"+item)
    return ls

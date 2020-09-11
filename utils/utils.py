import torch
from src.opt import opt
from config import config
import os
import matplotlib.pyplot as plt
import numpy as np


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
    title = ["model ID", "epoch", "lr", " ", "train_loss", "train_acc", "train_dist", "train_auc", "train_pr", "val_loss",
             "val_acc", "val_dist", "val_auc", "val_pr", " "]
    title += csv_body_part("train", "acc")
    title += csv_body_part("train", "dist")
    title += csv_body_part("train", "AUC")
    title += csv_body_part("train", "PR")
    title += csv_body_part("val", "acc")
    title += csv_body_part("val", "dist")
    title += csv_body_part("val", "AUC")
    title += csv_body_part("val", "PR")
    return title


def write_test_title():
    title = ["model ID", "params", "flops", "location", "inf_time", "test_loss", "test_acc", "test_dist", "test_auc",
             "test_pr", " "]
    title += csv_body_part("test", "acc")
    title += csv_body_part("test", "dist")
    title += csv_body_part("test", "AUC")
    title += csv_body_part("test", "PR")
    return title


def csv_body_part(phase, indicator):
    ls = []
    for item in config.body_part_name:
        ls.append(phase + "_" + item + "_" + indicator)
    ls.append(" ")
    return ls


def write_decay_title(num, char):
    char = char[:-1]
    for n in range(num):
        char += ",decay"
        char += str(n+1)
    char += "\n"
    return char


def write_decay_info(decays, char):
    char = char[:-1]
    for d in decays:
        char += ","
        char += str(d)
    char += "\n"
    return char


def draw_graph(epoch_ls, train_ls, val_ls, name, log_dir):
    ln1, = plt.plot(epoch_ls, train_ls, color='red', linewidth=3.0, linestyle='--')
    ln2, = plt.plot(epoch_ls, val_ls, color='blue', linewidth=3.0, linestyle='-.')
    plt.title("{}".format(name))
    plt.legend(handles=[ln1, ln2], labels=['train_{}'.format(name), 'val_{}'.format(name)])
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig(os.path.join(log_dir, "{}.jpg".format(name)))
    plt.cla()


# def draw_graph(epoch_ls, train_loss_ls, val_loss_ls, train_acc_ls, val_acc_ls, train_dists, val_dists, log_dir):
#     ln1, = plt.plot(epoch_ls, train_loss_ls, color='red', linewidth=3.0, linestyle='--')
#     ln2, = plt.plot(epoch_ls, val_loss_ls, color='blue', linewidth=3.0, linestyle='-.')
#     plt.title("Loss")
#     plt.legend(handles=[ln1, ln2], labels=['train_loss', 'val_loss'])
#     ax = plt.gca()
#     ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
#     ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
#     plt.savefig(os.path.join(log_dir, "loss.jpg"))
#     plt.cla()
#
#     ln1, = plt.plot(epoch_ls, train_acc_ls, color='red', linewidth=3.0, linestyle='--')
#     ln2, = plt.plot(epoch_ls, val_acc_ls, color='blue', linewidth=3.0, linestyle='-.')
#     plt.title("Acc")
#     plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
#     ax = plt.gca()
#     ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
#     ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
#     plt.savefig(os.path.join(log_dir, "acc.jpg"))
#     plt.cla()
#
#     ln1, = plt.plot(epoch_ls, train_dists, color='red', linewidth=3.0, linestyle='--')
#     ln2, = plt.plot(epoch_ls, val_dists, color='blue', linewidth=3.0, linestyle='-.')
#     plt.title("Dist")
#     plt.legend(handles=[ln1, ln2], labels=['train_dist', 'val_dist'])
#     ax = plt.gca()
#     ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
#     ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
#     plt.savefig(os.path.join(log_dir, "dist.jpg"))



def check_part(parts):
    tmp = []
    for part in parts:
        if np.sum((part > 0)) > 0:
            tmp.append(True)
        else:
            tmp.append(False)
    return np.array(tmp)


def check_hm(hms):
    tmp = []
    for hm in hms:
        if torch.sum(hm>0):
            tmp.append(True)
        else:
            tmp.append(False)
    return np.array(tmp)
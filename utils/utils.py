import torch
from config.opt import opt
from config import config
import os
import matplotlib.pyplot as plt


def init_model_list(folder):
    data_cfg, model_cfg, model, option = [], [], [], []
    for sub_folder in os.listdir(folder):
        sub_folder_path = os.path.join(folder, sub_folder)
        for file_name in os.listdir(sub_folder_path):
            model_cnt = 0
            file_path = os.path.join(sub_folder_path, file_name)
            if "data_cfg" in file_name:
                data_cfg.append(file_path)
            elif "model_cfg" in file_name:
                model_cfg.append(file_path)
            elif "option" in file_name:
                option.append(file_path)
            elif ".pth" in file_name:
                model.append(file_path)
                model_cnt += 1
                if model_cnt > 1:
                    raise AssertionError("More than one model exist in the folder: {}".format(sub_folder_path))
            else:
                continue
    assert len(model_cfg) == len(model) == len(option) == len(data_cfg), "Wrong length"
    return model, model_cfg, data_cfg, option


def get_superior_path(path):
    return "/".join(path.replace("\\", "/").split("/")[:-1])


def get_option_path(m_path):
    return os.path.join(get_superior_path(m_path), "option.pkl")


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


# def get_option_path(model_path):
#     model_path = model_path.replace("\\", "/")
#     option_path = "/".join(model_path.split("/")[:-1]) + "/option.pkl"
#     return option_path


# def adjust_lr(optimizer, epoch, lr_dict, nEpoch):
#     curr_ratio = epoch/nEpoch
#     bound = list(lr_dict.keys())
#     if curr_ratio > bound[0] and curr_ratio <= bound[1]:
#         lr = opt.LR * lr_dict[bound[0]]
#     elif curr_ratio > bound[1]:
#         lr = opt.LR * lr_dict[bound[1]]
#     else:
#         lr = opt.LR
#
#     for pg in optimizer.param_groups:
#         pg["lr"] = lr
#     return optimizer, lr
#
#
# def lr_decay(optimizer, lr):
#     lr = lr * 0.1
#     for pg in optimizer.param_groups:
#         pg["lr"] = lr
#     return optimizer, lr
#
#
# def warm_up_lr(optimizer, epoch):
#     bound = sorted(list(config.warm_up.keys()))
#     if epoch < bound[0]:
#         lr = opt.LR * config.warm_up[bound[0]]
#     elif epoch < bound[1]:
#         lr = opt.LR * config.warm_up[bound[1]]
#     else:
#         lr = opt.LR
#
#     for pg in optimizer.param_groups:
#         pg["lr"] = lr
#     return optimizer, lr
#
#
# def get_sparse_value():
#     if opt.epoch > opt.nEpochs * 0.5:
#         return opt.sparse_s * opt.sparse_decay
#     return opt.sparse_s


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
    title = ["model ID", "model name", "params", "flops", "inf_time", "location", "test_loss", "test_acc", "test_dist",
             "test_auc", "test_pr", " "]
    title += csv_body_part("test", "acc")
    title += csv_body_part("test", "dist")
    title += csv_body_part("test", "AUC")
    title += csv_body_part("test", "PR")
    title += csv_body_part("test", "thresh")
    return title


def csv_body_part(phase, indicator):
    ls = []
    body_parts = [item for item in config.body_parts.values()]
    if opt.kps == 17:
        body_parts = body_parts
    elif opt.kps == 13:
        body_parts = [body_parts[0]] + body_parts[5:]
    for item in body_parts:
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


# def check_part(parts):
#     tmp = []
#     for part in parts:
#         if np.sum((part > 0)) > 0:
#             tmp.append(True)
#         else:
#             tmp.append(False)
#     return np.array(tmp)
#
#
# def check_hm(hms):
#     tmp = []
#     for hm in hms:
#         if torch.sum(hm>0):
#             tmp.append(True)
#         else:
#             tmp.append(False)
#     return np.array(tmp)
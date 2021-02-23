import torch
from config import config


class Criterion:
    def build(self, crit, device="cuda:0"):
        if crit == "MSE":
            if device != "cpu":
                return torch.nn.MSELoss().cuda()
            else:
                return torch.nn.MSELoss()


class Optimizer:
    def build(self, optimize, params_to_update, lr, momentum, weightDecay):
        if optimize == 'rmsprop':
            optimizer = torch.optim.RMSprop(params_to_update, lr=lr, momentum=momentum,
                                            weight_decay=weightDecay)
        elif optimize == 'adam':
            optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=weightDecay)
        elif optimize == 'sgd':
            optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=momentum,
                                        weight_decay=weightDecay)
        else:
            raise Exception

        return optimizer


class StepLRScheduler:
    def __init__(self, epoch, warm_up_dict, decay_dict, lr):
        self.warm_up_dict = warm_up_dict
        self.warm_up_bound = sorted(list(warm_up_dict.keys()))
        self.decay_dict = decay_dict
        self.decay_bound = sorted(list(decay_dict.keys()))
        self.max_epoch = epoch
        self.base_lr = lr

    def update(self, optimizer, epoch):
        if epoch < max(self.warm_up_bound):
            lr = self.warm_up_lr(epoch)
        else:
            lr = self.decay_lr(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def decay_lr(self, epoch):
        lr = self.base_lr
        for bound in self.decay_bound[::-1]:
            if epoch > bound*self.max_epoch:
                lr = self.base_lr * self.decay_dict[bound]
                break
        return lr

    def warm_up_lr(self, epoch):
        lr = self.base_lr
        for bound in self.warm_up_bound:
            if epoch < bound:
                lr = self.base_lr * self.warm_up_dict[bound]
                break
        return lr


class SparseScheduler:
    def __init__(self, epochs, decay_dict, s):
        self.decay_dict = decay_dict
        self.decay_bound = sorted(list(decay_dict.keys()))
        self.max_epoch = epochs
        self.base_sparse = s

    def decay_sparse(self, epoch):
        s = self.base_sparse
        for bound in self.decay_bound[::-1]:
            if epoch > bound * self.max_epoch:
                s = s * self.decay_dict[bound]
                break
        return s

    def update(self, epoch):
        return self.decay_sparse(epoch)


def generate_cmd(ls):
    string = ""
    for idx, item in enumerate(ls):
        string += item
        string += " "
    return string[:-1] + "\n"


def summary_title():
    title_str = "id,backbone,structure,se ratio,DUC,input height,input width,params,flops,time,loss_weight,addDPG," \
                "kps,batch_size,optimizer,freeze_bn,freeze,sparse,total epochs,LR,Gaussian,thresh,weightDecay," \
                "loadModel,model_location, ,folder_name,training_time,train_acc,train_loss,train_dist,train_AUC," \
                "train_PR,val_acc,val_loss,val_dist,val_AUC,val_PR,best_epoch,final_epoch\n"
    return title_str


def csv_body_part(phase, indicator, kps=17):
    ls = []
    body_parts = [item for item in config.body_parts.values()]
    if kps == 17:
        body_parts = body_parts
    elif kps == 13:
        body_parts = [body_parts[0]] + body_parts[5:]
    for item in body_parts:
        ls.append(phase + "_" + item + "_" + indicator)
    ls.append(" ")
    return ls

def pckh_title(phase):
    ls = []
    for kps_name in ["left shoulder", "right shoulder", "left elbow", "right elbow", "left wrist", "right_wrist",
                     "left hip", "right hip", "left knee", "right knee", "left ankle", "right ankle"]:
        ls.append(phase + "_" + kps_name + "_pckh")
    ls.append(" ")
    return ls

def write_csv_title(kps=17):
    title = ["model ID", "epoch", "lr", " ", "train_loss", "train_acc", "train_dist", "train_auc", "train_pr", "val_loss",
             "val_acc", "val_dist", "val_auc", "val_pr", " "]
    title += csv_body_part("train", "acc", kps)
    title += pckh_title("train")
    title += csv_body_part("train", "dist", kps)
    title += csv_body_part("train", "AUC", kps)
    title += csv_body_part("train", "PR", kps)
    title += csv_body_part("val", "acc", kps)
    title += pckh_title("val")
    title += csv_body_part("val", "dist", kps)
    title += csv_body_part("val", "AUC", kps)
    title += csv_body_part("val", "PR", kps)
    return title


def warm_up_lr(optimizer, lr, epoch, warm_up_dict):
    bound = sorted(list(warm_up_dict.keys()))
    for b in bound:
        if epoch < b:
            lr = lr * warm_up_dict[b]

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return optimizer, lr

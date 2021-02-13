import torch


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
#-*-coding:utf-8-*-

from .datasets import BaseDataset
import torch


class TrainLoader:
    def __init__(self, data_info, data_cfg, joint_weight_dict=None):
        self.train_dataset = BaseDataset(data_info, data_cfg)
        self.val_dataset = BaseDataset(data_info, data_cfg, train=False)

    def build_dataloader(self, train_batch, val_batch, train_worker, val_worker, shuffle=True, pin_memory=True):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_batch, shuffle=shuffle, num_workers=train_worker, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=val_batch, shuffle=shuffle, num_workers=val_worker, pin_memory=pin_memory)
        return train_loader, val_loader


class TestLoader:
    def __init__(self, data_info, data_cfg):
        self.dataset = BaseDataset(data_info, data_cfg)

    def build_dataloader(self, batch, worker, shuffle=True, pin_memory=True):
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch, shuffle=shuffle, num_workers=worker,
                                             pin_memory=pin_memory)
        return loader

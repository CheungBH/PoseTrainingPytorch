from .datasets import MyDataset
import torch

open_source_dataset = ["coco"]


class TrainDataset:
    def __init__(self, data_info, joint_weight_dict=None, hmGauss=1, rotate=40):
        self.train_dataset = MyDataset(data_info, train=True, sigma=hmGauss, rot_factor=rotate)
        self.val_dataset = MyDataset(data_info, train=False, sigma=hmGauss, rot_factor=rotate)
        if self.is_shuffle(data_info):
            self.val_dataset.img_val, self.val_dataset.bbox_val, self.val_dataset.part_val = \
                self.train_dataset.img_val, self.train_dataset.bbox_val, self.train_dataset.part_val
        self.joint_weights = self.train_dataset.KP.unify_weighted(joint_weight_dict)

    def is_shuffle(self, info):
        for name, _ in info.items():
            if name not in open_source_dataset:
                return True
        return False

    def build_dataloader(self, train_batch, val_batch, train_worker, val_worker, shuffle=True, pin_memory=True):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=train_batch, shuffle=shuffle, num_workers=train_worker, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=val_batch, shuffle=shuffle, num_workers=val_worker, pin_memory=pin_memory)
        return train_loader, val_loader


class TestDataset:
    def __init__(self, data_info, hmGauss=1, rotate=40):
        self.dataset = MyDataset(data_info, train=True, sigma=hmGauss, rot_factor=rotate)

    def build_dataloader(self, batch, worker, shuffle=True, pin_memory=True):
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch, shuffle=shuffle, num_workers=worker, pin_memory=pin_memory)
        return loader

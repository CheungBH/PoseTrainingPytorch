# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import os
import h5py
from functools import reduce

import torch.utils.data as data
from src.opt import opt
import config.config as config
from utils.pose import generateSampleBox, choose_kps


origin_flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=config.sigma,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = config.train_data_path    # root image folders
        self.is_train = train           # training set or test set
        # self.inputResH = opt.inputResH
        # # print(os.path.isdir(self.img_folder))
        # self.inputResW = opt.inputResW
        # self.outputResH = opt.outputResH
        # self.outputResW = opt.outputResW
        self.inputResH = config.input_height
        self.inputResW = config.input_width
        self.outputResH = config.output_height
        self.outputResW = config.output_width
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.accIdxs = config.train_body_part
        self.flipRef = [item for idx, item in enumerate(origin_flipRef) if (idx+1)*2 < len(self.accIdxs)]

        # create train/val split
        with h5py.File(config.train_data_anno, 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:-5887] #:-5887
            self.bndbox_coco_train = annot['bndbox'][:-5887]
            self.part_coco_train = annot['part'][:-5887]
            # val
            self.imgname_coco_val = annot['imgname'][-5887:] #-5887:
            self.bndbox_coco_val = annot['bndbox'][-5887:]
            self.part_coco_val = annot['part'][-5887:]

        self.size_train = self.imgname_coco_train.shape[0]
        self.size_val = self.imgname_coco_val.shape[0]

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_coco_train[index]
            bndbox = self.bndbox_coco_train[index]
            imgname = self.imgname_coco_train[index]
        else:
            part = self.part_coco_val[index]
            bndbox = self.bndbox_coco_val[index]
            imgname = self.imgname_coco_val[index]

        imgname = reduce(lambda x, y: x + y,
                         map(lambda x: chr(int(x)), imgname))
        img_path = os.path.join(self.img_folder, imgname)

        part = choose_kps(part, self.accIdxs)

        metaData = generateSampleBox(img_path, bndbox, part, len(self.accIdxs),
                                     config.train_data, sf, self, train=self.is_train)

        inp, out, setMask = metaData
        return inp, out, setMask, config.train_data


    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val



import torch
import cv2
import os
import h5py
from functools import reduce
import numpy as np
import torch.utils.data as data
from src.opt import opt
import config.config as config
from utils.pose import generateSampleBox, choose_kps
import random
from dataset.bbox_visualize import BBoxVisualizer
from dataset.kps_visualize import KeyPointVisualizer
from utils.utils import check_hm, check_part
from src.opt import opt


origin_flipRef = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17))
open_source_dataset = config.open_source_dataset
draw = False


class Mscoco(data.Dataset):
    def __init__(self, data_path, train=True, val_img_num=5887, sigma=opt.hmGauss,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = data_path[0]    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.accIdxs = config.train_body_part
        self.flipRef = [item for idx, item in enumerate(origin_flipRef) if (idx+1)*2 < len(self.accIdxs)]

        # create train/val split
        with h5py.File(data_path[1], 'r') as annot:
            # train
            self.imgname_coco_train = annot['imgname'][:-val_img_num] #:-5887
            self.bndbox_coco_train = annot['bndbox'][:-val_img_num]
            self.part_coco_train = annot['part'][:-val_img_num]
            # val
            self.imgname_coco_val = annot['imgname'][-val_img_num:] #-5887:
            self.bndbox_coco_val = annot['bndbox'][-val_img_num:]
            self.part_coco_val = annot['part'][-val_img_num:]

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

        metaData = generateSampleBox(img_path, bndbox, part, len(self.accIdxs), config.train_data, sf, self,
                                     train=self.is_train)

        inp, out, setMask, (pt1, pt2) = metaData
        kps_info = (pt1, pt2, bndbox, cv2.imread(img_path), part)
        return inp, out, setMask, kps_info

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val


class MyDataset(data.Dataset):
    def __init__(self, data_info, train=True, sigma=opt.hmGauss,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.is_train = train  # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.accIdxs = config.train_body_part
        self.flipRef = [item for idx, item in enumerate(origin_flipRef) if (idx + 1) * 2 < len(self.accIdxs)]

        self.img_train, self.img_val, self.part_train, self.part_val, self.bbox_train, self.bbox_val = \
            [], [], [], [], [], []

        for k, v in data_info.items():
            if k in open_source_dataset:
                result = extract_data(v)
            else:
                result = extract_customized_data(v)
            self.img_train += result[0]
            self.bbox_train += result[1]
            self.part_train += result[2]
            self.img_val += result[3]
            self.bbox_val += result[4]
            self.part_val += result[5]

        self.size_train = len(self.img_train)
        self.size_val = len(self.img_val)

        self.img_train = np.array(self.img_train)
        self.bbox_train = np.array(self.bbox_train)
        self.part_train = np.array(self.part_train)
        self.img_val = np.array(self.img_val)
        self.bbox_val = np.array(self.bbox_val)
        self.part_val = np.array(self.part_val)

    def __getitem__(self, index):
        sf = self.scale_factor

        if self.is_train:
            part = self.part_train[index]
            bndbox = self.bbox_train[index]
            imgname = self.img_train[index]
        else:
            part = self.part_val[index]
            bndbox = self.bbox_val[index]
            imgname = self.img_val[index]
        part = choose_kps(part, self.accIdxs)

        inp, out, setMask, pt1, pt2 = generateSampleBox(imgname, bndbox, part, len(self.accIdxs), config.train_data, sf,
                                                        self, train=self.is_train)

        kps_info = (pt1, pt2, bndbox[0], imgname, part)
        return inp, out, setMask, kps_info

    def __len__(self):
        if self.is_train:
            return self.size_train
        else:
            return self.size_val


class TestDataset(MyDataset):
    def __init__(self, data_info,train=True, sigma=opt.hmGauss, scale_factor=(0.2, 0.3), rot_factor=40,
                         label_type='Gaussian'):
        super().__init__(data_info)
        if opt.kps == 17:
            self.accIdxs = [i + 1 for i in range(17)]
        elif opt.kps == 13:
            self.accIdxs = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def extract_customized_data(data_info):
    data_folder, h5file, val_num = data_info[0], data_info[1], data_info[2]
    with h5py.File(h5file, 'r') as annot:
        imgname = annot['imgname'][:].tolist()  #:-5887
        bndbox_raw = annot['bndbox'][:].tolist()
        bndbox = [[xywh2xyxy(box[0])] for box in bndbox_raw]
        part = annot['part'][:].tolist()

        imgs = []
        for i in imgname:
            i = change_imgname(i)
            imgs.append(os.path.join(data_folder, reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), i))))

    val_ls = random.sample(range(len(imgs)), val_num)

    if draw:
        BBV, KPV = BBoxVisualizer(), KeyPointVisualizer()
        for idx, (im_name, box, p) in enumerate(zip(imgs, bndbox, part)):
            img = cv2.imread(im_name)
            img = BBV.visualize(box, img)
            img = KPV.vis_ske(img, torch.FloatTensor([p]), KPV.scoredict2tensor(1))
            img = cv2.resize(img, (720, 540))
            cv2.imshow("res", img)
            cv2.waitKey(0)
            a = 1

    img_train, bbox_train, part_train, img_val, bbox_val, part_val = [], [], [], [], [], []
    for i, (im, bbx, pt) in enumerate(zip(imgs, bndbox, part)):
        if i not in val_ls:
            img_train.append(im)
            bbox_train.append(bbx)
            part_train.append(pt)
        else:
            img_val.append(im)
            bbox_val.append(bbx)
            part_val.append(pt)
    
    return [img_train, bbox_train, part_train, img_val, bbox_val, part_val]


def extract_data(data_info):
    data_folder, h5file, val_num = data_info[0], data_info[1], data_info[2]
    with h5py.File(h5file, 'r') as annot:
        imgname_train = annot['imgname'][:-val_num].tolist()  #:-5887
        bndbox_train = annot['bndbox'][:-val_num].tolist()
        part_train = annot['part'][:-val_num].tolist()
        # val
        imgname_val = annot['imgname'][-val_num:].tolist()  # -5887:
        bndbox_val = annot['bndbox'][-val_num:].tolist()
        part_val = annot['part'][-val_num:].tolist()

        img_train, img_val = [], []
        for i in imgname_train:
            # imgname = change_imgname(i)
            img_train.append(os.path.join(data_folder, reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), i))))

        for i in imgname_val:
            # imgname = change_imgname(i)
            img_val.append(os.path.join(data_folder, reduce(lambda x, y: x + y, map(lambda x: chr(int(x)), i))))

    return [img_train, bndbox_train, part_train, img_val, bndbox_val, part_val]


def xywh2xyxy(box):
    return [box[0], box[1], box[0]+box[2], box[1]+box[3]]


def change_imgname(img_name):
    temp = np.array([])
    for item in img_name:
        if item != -1:
            temp = np.append(temp, item)
    return temp


if __name__ == '__main__':
    print(xywh2xyxy([1,2,1,2]))

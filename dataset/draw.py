#-*-coding:utf-8-*-
import numpy as np
import cv2
import torch
import math
from dataset.visualize import KeyPointVisualizer

tensor = torch.Tensor


class PredictionVisualizer:
    def __init__(self, kps, bs, out_height, out_width, in_height, in_width, max_img=4, column=4, final_size=(720, 540),
                 dataset="coco"):
        self.kps = kps
        self.img_num = min(bs, max_img)
        self.column = column
        self.row = math.floor(self.img_num / column)
        self.final_size = final_size
        self.out_width = out_width
        self.out_height = out_height
        self.in_width = in_width
        self.in_height = in_height
        self.KPV = KeyPointVisualizer(self.kps, dataset)

    def getPred(self, hm):
        max_val = 0
        pred = [0, 0]
        for column in range(hm.size(0)):
            for row in range(hm.size(1)):
                if hm[column][row] > max_val:
                    max_val = hm[column][row]
                    pred = [row, column]
        return pred, max_val

    def draw_kps(self, hms, meta_data):
        img_path = meta_data["name"]
        padded_size = meta_data["padded_size"]
        box = meta_data["enlarged_box"]
        img = cv2.imread(img_path)
        # img_h, img_w = img.shape[0], img.shape[1]
        img_h, img_w = box[3] - box[1], box[2] - box[0]
        kps, kps_score = [], []
        resize_ratio = min(self.in_width / img_w, self.in_height / img_h)
        for i in range(hms.size(0)):
            max_location, max_val = self.getPred(hms[i])
            kps_score.append([max_val])
            x_coord = (max_location[0]/self.out_width * self.in_width - padded_size[0])/resize_ratio + box[0]
            y_coord = (max_location[1]/self.out_height * self.in_height - padded_size[1])/resize_ratio + box[1]
            kps.append([x_coord, y_coord])
        self.KPV.visualize(img, [kps], [kps_score])
        return cv2.resize(img, self.final_size)

    def process(self, hms, img_metas):
        img_ls = []
        for i, hm in enumerate(hms):
            if i >= self.img_num:
                break

            img_ls.append(self.draw_kps(hm, self.extract_current_meta(i, img_metas)))
        if self.img_num == 1:
            return img_ls[0]

        prediction_1 = np.concatenate((img_ls[0], img_ls[1]), axis=0)
        prediction_2 = np.concatenate((img_ls[2], img_ls[3]), axis=0)
        predictions = np.concatenate((prediction_1, prediction_2), axis=1)
        return predictions

    @staticmethod
    def extract_current_meta(idx, meta):
        tmp = {}
        tmp["enlarged_box"] = meta["enlarged_box"][idx]
        tmp["name"] = meta["name"][idx]
        tmp["padded_size"] = meta["padded_size"][idx]
        return tmp


class HeatmapVisualizer:
    def __init__(self, height, width, hm_column=6):
        self.height = height
        self.width = width
        self.hm_column = hm_column

    def draw_hms(self, hms):
        hms = hms.cpu().numpy()
        hm1 = np.concatenate((hms[0], hms[1], hms[2], hms[3], hms[4], hms[5]), axis=1)
        hm2 = np.concatenate((hms[6], hms[7], hms[8], hms[9], hms[10], hms[11]), axis=1)
        remain = len(hms) - self.hm_column * 2
        hm3 = self.generate_white(self.hm_column-remain)
        for num in range(remain):
            hm3 = np.concatenate((hms[-num+1], hm3), axis=1)
        hm = np.concatenate((hm1, hm2, hm3), axis=0)
        return tensor(hm).unsqueeze(dim=0)

    def generate_white(self, num):
        rand = np.zeros((self.height, self.width))
        for _ in range(num - 1):
            rand = np.concatenate((rand, np.zeros((self.height, self.width))), axis=1)
        return rand

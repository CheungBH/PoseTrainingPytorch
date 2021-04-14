#-*-coding:utf-8-*-
import numpy as np
import cv2
import torch
import math
import os
from dataset.visualize import KeyPointVisualizer

tensor = torch.Tensor


class PredictionVisualizer:
    def __init__(self, kps, bs, max_img=4, column=4, hm_colomn=6):
        self.kps = kps
        self.img_num = min(bs, max_img)
        self.column = column
        self.row = math.floor(self.img_num / column)
        self.hm_column = hm_colomn

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

    def visualize(self, hms, img_meta, result_path=""):
        img = cv2.imread(img_meta["img_path"])
        self.inp_height, self.inp_width = img.shape[0], img.shape[1]
        self.out_height, self.out_width = hms[0].size()
        hms = hms.cpu()
        img_ls = []
        for i in range(self.img_num):
            pt1, pt2, boxes, img_path, hm = _pt1[i].unsqueeze(dim=0), _pt2[i].unsqueeze(dim=0), _boxes[i].unsqueeze(
                dim=0), _img_path[i], hms[i].unsqueeze(dim=0)
            img, if_kps = self.draw_kp(hm, pt1, pt2, boxes, img_path)
            img = cv2.resize(img, (720, 540))
            img_ls.append(img)

        prediction_1 = np.concatenate((img_ls[0], img_ls[1]), axis=0)
        prediction_2 = np.concatenate((img_ls[2], img_ls[3]), axis=0)
        predictions = np.concatenate((prediction_1, prediction_2), axis=1)

        cv2.imwrite(os.path.join(result_path, "logs", "img.jpg"), predictions)

        return predictions

    def generate_white(self, num):
        rand = np.zeros((self.out_height, self.out_width))
        for _ in range(num - 1):
            rand = np.concatenate((rand, np.zeros((self.out_height, self.out_width))), axis=1)
        return rand

    def draw_kp(self, hm, pt1, pt2, boxes, img_path):
        scores = tensor([[0.999]] * (boxes.shape[0]))
        boxes = boxes.float()
        preds_hm, preds_img, preds_scores = getPrediction(
            hm, pt1, pt2, self.inp_height, self.inp_width, self.out_height, self.out_width)
        kps, score = pose_nms(boxes, scores, preds_img, preds_scores, pose_classes)
        orig_img = cv2.imread(img_path)
        if kps:
            cond = True
            kpv = KeyPointVisualizer()
            img = kpv.vis_ske(orig_img, kps, score)
        else:
            img = orig_img
            cond = False
        return img, cond



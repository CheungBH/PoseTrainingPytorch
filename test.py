from utils.eval import getPrediction
from utils.nms import pose_nms
import torch
import cv2
from config import config
from utils.visualize import KeyPointVisualizer
from src.opt import opt
import math
import numpy as np

tensor = torch.Tensor
max_img = 4
img_num = min(opt.validBatch, max_img)
column = 4
row = math.floor(img_num/column)


def draw_kp(hm, pt1, pt2, boxes, img_path):
    scores = tensor([[0.999]]*(boxes.shape[0]))
    preds_hm, preds_img, preds_scores = getPrediction(
        hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
    kps, score = pose_nms(boxes, scores, preds_img, preds_scores)
    orig_img = cv2.imread(img_path)
    if kps:
        cond = True
        kpv = KeyPointVisualizer()
        img = kpv.vis_ske(orig_img, kps, score)
        # cv2.imwrite("img.jpg", img)
    else:
        img = orig_img
        cond = False
    return img, cond


def draw_kps(hms, info):
    hms = hms.cpu()
    (_pt1, _pt2, _boxes, _img_path) = info
    drawn = False
    img_ls = []
    for i in range(img_num):
        pt1, pt2, boxes, img_path, hm = _pt1[i].unsqueeze(dim=0), _pt2[i].unsqueeze(dim=0), _boxes[i].unsqueeze(dim=0), _img_path[i], hms[i].unsqueeze(dim=0)
        img, if_kps = draw_kp(hm, pt1, pt2, boxes, img_path)
        drawn = if_kps or drawn
        img = cv2.resize(img, (720, 540))
        img_ls.append(img)

    # predictions = img_ls[0]
    prediction_1 = np.concatenate((img_ls[0], img_ls[1]), axis=0)
    prediction_2 = np.concatenate((img_ls[2], img_ls[3]), axis=0)
    predictions = np.concatenate((prediction_1, prediction_2), axis=1)
    # for r in range(row):
    #     row_img = [img_ls[r*4+c] for c in range(column)]
    #     predictions = np.concatenate((row_img[0], row_img[1], row_img[2], row_img[3]), axis=0)
    cv2.imwrite("img.jpg", predictions)

    return predictions, drawn



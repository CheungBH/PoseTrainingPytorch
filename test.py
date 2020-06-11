from utils.eval import getPrediction
from utils.nms import pose_nms
import torch
import cv2
from config import config
from utils.visualize import KeyPointVisualizer
from src.opt import opt
import math
import numpy as np

tensor = torch.FloatTensor
max_img = 4
img_num = min(opt.validBatch, max_img)
column = 4
row = math.floor(img_num/column)


def draw_kp(hm, pt1, pt2, boxes, img_path):
    scores = tensor([[0.999]]*(boxes.shape[0]))
    boxes = boxes.float()
    preds_hm, preds_img, preds_scores = getPrediction(
        hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
    kps, score = pose_nms(boxes, scores, preds_img, preds_scores)
    orig_img = cv2.imread(img_path)
    if not kps:
        cond = True
        kpv = KeyPointVisualizer()
        img = kpv.vis_ske(orig_img, kps, score)
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


def draw_hms(hms):
    hm_column = 6
    hms = hms.cpu().numpy()
    hm1 = np.concatenate((hms[0], hms[1], hms[2], hms[3], hms[4], hms[5]), axis=1)
    hm2 = np.concatenate((hms[6], hms[7], hms[8], hms[9], hms[10], hms[11]), axis=1)
    remain = len(hms) - hm_column * 2
    hm3 = generate_white(hm_column-remain)
    for num in range(remain):
        hm3 = np.concatenate((hms[-num+1], hm3), axis=1)
    hm = np.concatenate((hm1, hm2, hm3), axis=0)
    return tensor(hm).unsqueeze(dim=0)


def generate_white(num):
    rand = np.zeros((opt.outputResH, opt.outputResW))
    for _ in range(num-1):
        rand = np.concatenate((rand, np.zeros((opt.outputResH, opt.outputResW))), axis=1)
    return rand

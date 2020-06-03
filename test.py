from utils.eval import getPrediction
from utils.nms import pose_nms
import torch
import cv2
from config import config
from utils.visualize import KeyPointVisualizer

tensor = torch.Tensor


def draw_kps(hms, info):
    (pt1, pt2, boxes, img_path) = info
    scores = tensor([0.999]*(boxes.shape[0]))
    orig_img = cv2.imread(img_path)
    preds_hm, preds_img, preds_scores = getPrediction(
        hms, pt1, pt2, config.input_height, config.input_width, config.output_height, config.output_width)
    kps, score = pose_nms(boxes, scores, preds_img, preds_scores)
    kpv = KeyPointVisualizer()
    img = kpv.vis_ske(orig_img, kps, score)
    return img


#-*-coding:utf-8-*-
import numpy as np
import cv2
import torch
import math
from dataset.visualize import KeyPointVisualizer

tensor = torch.Tensor


class HeatmapPredictor:
    def __init__(self, out_height, out_width, in_height, in_width):
        self.out_width = out_width
        self.out_height = out_height
        self.in_width = in_width
        self.in_height = in_height

    def getPrediction(self, hms):
        '''
        Get keypoint location from heatmaps
        '''

        assert hms.dim() == 4, 'Score maps should be 4-dim'
        maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

        maxval = maxval.view(hms.size(0), hms.size(1), 1)
        idx = idx.view(hms.size(0), hms.size(1), 1) + 1

        preds = idx.repeat(1, 1, 2).float()

        preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))
        return preds, maxval

    def revert_locations(self, locations, padded_size, resize_ratio, box):
        resumed_locations = torch.zeros_like(locations)
        for idx in range(len(locations)):
            resumed_locations[idx][0] = (locations[idx][0]/self.out_width * self.in_width - padded_size[0])/resize_ratio + box[0]
            resumed_locations[idx][1] = (locations[idx][1]/self.out_height * self.in_height - padded_size[1])/resize_ratio + box[1]
        return resumed_locations

    def decode_hms(self, hms, meta_datas):
        # locations, max_vals = [], []
        location_whole = []
        locations, max_vals = self.getPrediction(hms)
        for meta_data, location in zip(meta_datas, locations):
            padded_size = meta_data["padded_size"]
            box = meta_data["enlarged_box"]
            # img_h, img_w = img.shape[0], img.shape[1]
            img_h, img_w = box[3] - box[1], box[2] - box[0]
            resize_ratio = min(self.in_width / img_w, self.in_height / img_h)
            location_whole.append(self.revert_locations(location, padded_size, resize_ratio, box).tolist())
        return torch.tensor(location_whole), torch.tensor(max_vals)


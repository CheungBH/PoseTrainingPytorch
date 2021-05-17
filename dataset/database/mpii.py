import json
import os
from .base import *
import torch


class MPII(BaseDataset):
    def __init__(self, kps):
        super().__init__(kps)
        self.kps_num = 16
        self.convert_13_idx = [6,10,9,11,8,12,7,3,2,4,1,5,0]

    def init_kps(self):
        self.KPP.init_kps(self.kps, "mpii")
        return self.KPP.get_kps_info()

    def load_data(self, json_file, folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        kps_valid = []
        img_names = {}
        for i in range(len(anno['images'])):
            res = anno["images"][i]
            img_names[res["id"]] = os.path.join(folder_name, str(res['file_name']))
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            if not sum(kp_valid):
                continue
            bbox.append(xywh2xyxy(entry['bbox']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            images.append(img_names[entry["image_id"]])
        if self.kps == 13:
            keypoint = self.convert_13kps(keypoint)
            kps_valid = self.convert_13kps_valid(kps_valid)
        return images, keypoint, bbox, ids, kps_valid

    def convert_13kps(self, item):
        tns = torch.Tensor(item)
        nose_tns = (tns[:,8,:] + tns[:,9,:])/2
        kps13_tns = torch.cat((tns[:,0:6,:], nose_tns.unsqueeze(dim=1), tns[:,10:16,:]), dim=1)
        sorted_tns = kps13_tns.permute(1,0,2)[self.convert_13_idx].permute(1,0,2)
        return sorted_tns.tolist()

    def convert_13kps_valid(self, item):
        tns = torch.Tensor(item)
        nose_tns = (tns[:,8] + tns[:,9])/2
        kps13_tns = torch.cat((tns[:,0:6], nose_tns.unsqueeze(dim=1), tns[:,10:16]), dim=1)
        sorted_tns = kps13_tns.permute(1,0)[self.convert_13_idx].permute(1,0)
        return sorted_tns.tolist()

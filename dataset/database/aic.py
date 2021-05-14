import json
import os
from .base import *
import torch


class AIChallenger(BaseDataset):
    def __init__(self, kps):
        super().__init__(kps)
        self.kps_num = 14
        # self.convert_13_idx = [2, 4, 6, 1, 3, 5, 8, 10, 12, 7, 9, 11, 0]
        self.convert_13_idx = [12,3,0,4,1,5,2,9,6,10,7,11,8]

    def init_kps(self):
        self.KPP.init_kps(self.kps, "aic")
        return self.KPP.get_kps_info()

    def load_data(self, json_file, folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        kps_valid = []
        for i in range(len(anno)):
            images.append(os.path.join(folder_name,str(anno[i]['image_id'])+'.jpg'))
            kp, kp_valid = kps_reshape(anno[i]["keypoint_annotations"]["human1"])
            if not sum(kp_valid):
                continue
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            bbox.append(anno[i]['human_annotations']["human1"])
            ids.append(anno[i]['url'])
        if self.kps == 13:
            keypoint = self.convert_13kps(keypoint)
            kps_valid = self.convert_13kps_valid(kps_valid)
        return images, keypoint, bbox, ids, kps_valid

    def convert_13kps(self, item):
        tns = torch.Tensor(item)
        nose_tns = (tns[:,-1,:] + tns[:,-2,:])/2
        kps12_tns = tns[:,0:12,:]
        kps13_tns = torch.cat((kps12_tns, nose_tns.unsqueeze(dim=1)), dim=1)
        sorted_tns = kps13_tns.permute(1,0,2)[self.convert_13_idx].permute(1,0,2)
        return sorted_tns.tolist()

    def convert_13kps_valid(self, item):
        tns = torch.Tensor(item)
        nose_tns = (tns[:,-1] + tns[:,-2])/2
        kps12_tns = tns[:,0:12]
        kps13_tns = torch.cat((kps12_tns, nose_tns.unsqueeze(dim=1)), dim=1)
        sorted_tns = kps13_tns.permute(1,0)[self.convert_13_idx].permute(1,0)
        return sorted_tns.tolist()

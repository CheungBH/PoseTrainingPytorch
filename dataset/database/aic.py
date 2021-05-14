import json
import os
from .base import *


class AIChallenger(BaseDataset):
    def __init__(self, kps):
        super().__init__(kps)
        self.kps_num = 13

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
            kp, kp_valid = select_kps(kp, kp_valid, self.body_part_idx, self.kps_num)
            if not sum(kp_valid):
                continue
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            bbox.append(anno[i]['human_annotations']["human1"])
            ids.append(anno[i]['url'])
        return images, keypoint, bbox, ids, kps_valid

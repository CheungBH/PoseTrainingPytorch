import json
import os
from .base import *


class CEILING(BaseDataset):
    def __init__(self, kps, phase):
        super().__init__(kps, phase)
        self.kps_num = 17

    def init_kps(self):
        self.KPP.init_kps(self.kps, "yoga")
        return self.KPP.get_kps_info()

    def load_data(self, json_file,folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        kps_valid = []
        # for i in range(len(anno['images'])):
        #     images.append(os.path.join(folder_name,str(anno['images'][i]['file_name'])))
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["image_id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            if not sum(kp_valid):
                continue
            if len(kp) == 16:
                # images.pop(i)
                continue
            images.append(os.path.join(folder_name, entry["image_id"]+'.jpg'))
            bbox.append(xywh2xyxy(entry['bbox']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
        return images, keypoint, bbox, ids, kps_valid



import json
import os
from .base import *


class Ball(BaseDataset):
    def __init__(self, kps, phase):
        super().__init__(kps, phase)
        self.kps_num = 1

    def init_kps(self):
        self.KPP.init_kps(self.kps, "ball")
        return self.KPP.get_kps_info()

    def load_data(self, json_file, folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        images_res = []
        kps_valid = []
        for i in range(len(anno['images'])):
            images_res.append(anno['images'][i]['file_name'])
        for img_info in anno['annotations']:
            kp, kp_valid = kps_reshape(img_info["keypoints"])
            kp, kp_valid = select_kps(kp, kp_valid, self.body_part_idx, self.kps_num)
            if not sum(kp_valid):
                continue

            #images.append(os.path.join(folder_name, str(img_info['image_id']).zfill(12) + ".jpg"))
            images.append(os.path.join(folder_name, anno['images'][img_info['image_id']]['file_name']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            ids.append(img_info["id"])
            bbox.append(xywh2xyxy(img_info['bbox']))
        return images, keypoint, bbox, ids, kps_valid



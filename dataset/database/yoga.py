import json
import os
from .base import *


class YOGA(BaseDataset):
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
        # images_res = []
        kps_valid = []
        # for i in range(len(anno['images'])):
        #     images_res.append(anno['images'][i]['file_name'])
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            kp, kp_valid = select_kps(kp, kp_valid, self.body_part_idx, self.kps_num)
            if not sum(kp_valid):
                continue
            bbox.append(xywh2xyxy(entry['bbox']))
            if self.phase == "valid":
                images.append(os.path.join(folder_name, str(entry['image_id']) + ".jpg"))
            else:
                images.append(os.path.join(folder_name, str(entry['image_id']).zfill(12)))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
        return images, keypoint, bbox, ids, kps_valid



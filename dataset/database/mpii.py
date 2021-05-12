import json
import os
from .base import *


class MPII(BaseDataset):
    def __init__(self, kps):
        super().__init__(kps)
        self.kps_num = 14

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
            ids.append(entry["image_id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            kp, kp_valid = select_kps(kp, kp_valid, self.body_part_idx, self.kps_num)
            if not sum(kp_valid):
                continue
            bbox.append(xywh2xyxy(entry['bbox']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
        name = list(img_names.keys())
        value = list(img_names.values())
        num = 0
        # for i in range(len(ids)):
        #     if name[num] == ids[i]:
        #         images.append(value[i])
        #         num += 1
        #     else:
        #         images.append()
        #         num -= 1
        return images, keypoint, bbox, ids, kps_valid


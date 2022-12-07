import json
import os
from .base import *
import numpy as np

class Thermal(BaseDataset):
    def __init__(self, kps, phase):
        super().__init__(kps, phase)
        self.kps_num = 17

    def init_kps(self):
        self.KPP.init_kps(self.kps, "coco")
        return self.KPP.get_kps_info()

    def load_data(self, json_file, folder_name):
        images, keypoint, bbox, ids, kps_valid = self.json_preprocessing(json_file,folder_name)
        return images, keypoint, bbox, ids, kps_valid

    def json_preprocessing(self, json_file,folder_name):
        anno = json.load(open(json_file))
        images = []
        keypoint = []
        kps_valid = []
        id = []
        boxes = []
        for annotation in anno:
            visible = annotation['visible']
            img_name = annotation['img'][24:]
            id_temp = annotation['img'][15:23]
            for person_id in range(len(visible)//15):
                keypoint_temp = np.zeros([13,2])
                kps_valid_temp = np.zeros([13,1])
                k = 0
                for i in range(15):
                    if i == 1:
                        keypoint_temp[k,0] = (visible[i-1]['x'] + visible[i]['x'])/2
                        keypoint_temp[k,1] = (visible[i-1]['y'] + visible[i]['y'])/2
                        kps_valid_temp[k,0] = 0 if visible[i]['number'] == 3 else 1
                        k = k + 1
                    elif i == 8:
                        pass
                    elif i == 0:
                        pass
                    else:
                        keypoint_temp[k,0] = visible[i]['x']
                        keypoint_temp[k,1] =  visible[i]['y']
                        kps_valid_temp[k,0] = 0 if visible[i]['number'] == 3 else 1
                        k = k + 1

                keypoint_temp = keypoint_temp / 100
                keypoint_temp[:,0] = keypoint_temp[:,0] * visible[0]['original_width']
                keypoint_temp[:,1] = keypoint_temp[:,1] * visible[0]['original_height']
                minus_factor = 0.2
                add_factor = 0.2
                xmax = np.amax(keypoint_temp[:,0])
                xmin = np.amin(keypoint_temp[:,0])
                ymax = np.amax(keypoint_temp[:,1])
                ymin = np.amin(keypoint_temp[:,1])
                box_temp = np.array([xmin - (minus_factor*(xmax-xmin)), \
                                     ymin - (minus_factor*(ymax-ymin)),\
                                     xmax + (add_factor*(xmax-xmin)),\
                                     ymax + (add_factor*(ymax-ymin))])

                keypoint.append(keypoint_temp)
                kps_valid.append(kps_valid_temp)
                images.append(os.path.join(folder_name,img_name))
                id.append(id_temp)
                boxes.append(box_temp)

        return images, keypoint, boxes, id, kps_valid


#images, keypoint, kps_valid, id = Thermal.json_preprocessing('/media/hkuit164/Backup/project-1-at-2022-08-31-03-03-2c12cb06.json')

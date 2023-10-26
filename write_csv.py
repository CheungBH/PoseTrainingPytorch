import sys

from models.pose_model import PoseModel
from dataset.transform import ImageTransform
from dataset.draw import PredictionVisualizer
from utils.utils import get_option_path
import cv2
import os
from utils.utils import get_corresponding_cfg
import torch
import csv


posenet = PoseModel()

class ImageVisualizer:
    out_h, out_w, in_h, in_w = 64, 64, 256, 256

    def __init__(self, model_cfg, model_path, data_cfg=None, show=True, device="cuda", conf=0.05):
        self.show = show
        self.device = device
        option_file = get_option_path(model_path)
        self.transform = ImageTransform()

        if os.path.exists(option_file):
            option = torch.load(option_file)
            self.out_h, self.out_w, self.in_h, self.in_w = \
                option.output_height, option.output_width, option.input_height, option.input_width
        else:
            if data_cfg:
                self.transform.init_with_cfg(data_cfg)
                self.out_h, self.out_w, self.in_h, self.in_w = \
                    self.transform.output_height, self.transform.output_width, self.transform.input_height,self.transform.input_width
            else:
                pass

        posenet.build(model_cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        self.model.eval()
        self.conf = conf
        posenet.load(model_path)
        self.PV = PredictionVisualizer(posenet.kps, 1, self.out_h, self.out_w, self.in_h, self.in_w, max_img=1, column=1)

    def visualize(self, img_path):
        with torch.no_grad():
            img = cv2.imread(img_path)
            inp, padded_size = self.transform.process_single_img(img_path, self.out_h, self.out_w, self.in_h, self.in_w)
            img_meta = {
                "name": img_path,
                "enlarged_box": [0, 0, img.shape[1], img.shape[0]],
                "padded_size": padded_size
            }

            if self.device != "cpu":
                inp = inp.cuda()
            out = self.model(inp.unsqueeze(dim=0))
            location, img_h, img_w = self.PV.draw_kps_csv(out, img_meta, self.conf)
            # flatten_array = location.flatten()
            # processed_string = str(flatten_array).replace("tensor([", "").replace("], device='cuda:0')", "").replace("\n", "").replace("       ", "")
            float_numbers = [float(i) for i in location.flatten().tolist()]#[float(num) for num in processed_string.split(",")]
            modified_array = []
            for index, num in enumerate(float_numbers):
                if index % 2 == 0:
                    modified_array.append(num / img_w)
                else:
                    modified_array.append(num / img_h)

            modified_array.extend([2, "Fall", filename])
            # print(modified_array)
            csv_path = '/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/1_Fall.csv'
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(modified_array)

if __name__ == '__main__':
    model_path = "/home/hkuit164/Desktop/xjl/1025+1103+1121/alphapose/latest.pth"
    folder_path = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/fall/project-40-at-2023-10-24-09-20-7576869d/Fall"
    conf = 0.05

    model_cfg = ""
    data_cfg = ""

    if not model_path or not data_cfg:
        model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])
    IV = ImageVisualizer(model_cfg, model_path, data_cfg, conf=conf)
    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            try:
                IV.visualize(img_path)
            except:
                print(idx)
                sys.exit(1)

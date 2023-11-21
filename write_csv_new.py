import sys

from models.pose_model import PoseModel
from dataset.transform import ImageTransform
from dataset.draw import PredictionVisualizer
from utils.utils import get_option_path
from PIL import Image
import cv2
import os
from utils.utils import get_corresponding_cfg
import torch
import csv
import json


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
        if isinstance(conf, float):
            self.conf = torch.tensor([conf for _ in range(self.kps)])
        else:
            self.conf = torch.tensor([float(i) for i in conf.split(",")])
        self.conf = conf
        posenet.load(model_path)
        self.PV = PredictionVisualizer(posenet.kps, 1, self.out_h, self.out_w, self.in_h, self.in_w, max_img=1, column=1)

    def img_splitcate(self, json_path, scale_factor, folder_path, cate_output):
        with open(json_path, "r") as f:
            data = json.load(f)

        image_files = os.listdir(folder_path)

        os.makedirs(cate_output, exist_ok=True)

        category_count = {}

        for image_data in data["images"]:
            image_id = image_data["id"]
            file_name = image_data["file_name"]

            if file_name in image_files:
                image_path = os.path.join(folder_path, file_name)

                annotation = next((ann for ann in data["annotations"] if ann["image_id"] == image_id), None)
                if annotation:
                    category_id = annotation["category_id"]
                    bbox = annotation["bbox"]

                    category_name = next((cat["name"] for cat in data["categories"] if cat["id"] == category_id), None)
                    if category_name:

                        category_folder = os.path.join(cate_output, category_name)
                        os.makedirs(category_folder, exist_ok=True)

                        if category_name in category_count:
                            category_count[category_name] += 1
                        else:
                            category_count[category_name] = 1

                        new_file_name = f"{os.path.splitext(file_name)[0]}_{category_name}_{category_count[category_name]}.jpg"

                        image = Image.open(image_path)

                        scale_factor /= 2
                        width_scale = bbox[2] * scale_factor
                        height_scale = bbox[3] * scale_factor

                        left = max(0, bbox[0] - width_scale)
                        top = max(0, bbox[1] - height_scale)
                        right = min(image.width, bbox[0] + bbox[2] + width_scale)
                        bottom = min(image.height, bbox[1] + bbox[3] + height_scale)

                        if left < right and top < bottom:
                            cropped_image = image.crop((left, top, right, bottom))

                            cropped_image.save(os.path.join(category_folder, new_file_name))
                        else:
                            print(f"Invalid coordinates for cropping: {bbox}")
                    else:
                        print(f"Cannot find category name for category_id: {category_id}")
                else:
                    print(f"No annotation found for image_id: {image_id}")
            else:
                print(f"Image file not found: {file_name}")

    def visualize(self, img_path, label_file, csv_path, folder_path):
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

            max_value = self.PV.getPrediction(out)[1]
            if_exist = [(v>c).tolist() for c, v in zip(self.conf, max_value.squeeze())]

            float_numbers = [float(i) for i in location.flatten().tolist()]
            modified_array = []
            for index, num in enumerate(float_numbers):
                if if_exist[int(index/2)] is True:
                    if index % 2 == 0:
                        modified_array.append(num / img_w)
                    else:
                        modified_array.append(num / img_h)
                else:
                    modified_array.append(-1)

            with open(label_file, 'r') as label:
                cate_array = label.readlines()

            folder_cate = os.path.basename(folder_path)

            for idx, cate in enumerate(cate_array):
                if cate[:-1] == folder_cate:
                    modified_array.extend([f"{idx}", f"{cate[:-1]}", filename])
                    # print(modified_array)
                    with open(csv_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(modified_array)

if __name__ == '__main__':
    # Pose model path
    model_path = "/home/hkuit164/Desktop/xjl/1025+1103+1121/alphapose/latest.pth"
    folder_path = "/media/hkuit164/Backup/xjl/ML_data_process/Label_Studio/result/val/images"
    json_path = "/media/hkuit164/Backup/xjl/ML_data_process/Label_Studio/result/val/result.json"
    label_path = "/media/hkuit164/Backup/xjl/ML_data_process/ML/csv/label"

    # output
    csv_path = "/media/hkuit164/Backup/xjl/ML_data_process/ML/csv/test.csv"
    cate_output = "/media/hkuit164/Backup/xjl/ML_data_process/ML/csv/test"

    conf = 0.05

    model_cfg = ""
    data_cfg = ""
    option_path = ""

    if not model_path or not data_cfg:
        model_cfg, data_cfg, option_path = get_corresponding_cfg(model_path, check_exist=["data", "model"])

    if os.path.exists(option_path):
        info = torch.load(option_path)
        if "thresh" in info:
            conf = info.thresh

    IV = ImageVisualizer(model_cfg, model_path, data_cfg, conf=conf)

    with open(data_cfg, "r") as load_f:
        load_dict = json.load(load_f)
    scale_factor = load_dict["scale"]

    IV.img_splitcate(json_path, scale_factor, folder_path, cate_output)

    for img_folder_name in os.listdir(cate_output):
        img_folder_path = os.path.join(cate_output, img_folder_name)
        if os.path.isdir(img_folder_path):
            for idx, filename in enumerate(os.listdir(img_folder_path)):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(img_folder_path, filename)
                    try:
                        IV.visualize(img_path, label_path, csv_path, img_folder_path)
                    except:
                        print(idx)
                        sys.exit(1)

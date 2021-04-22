from torchvision.transforms import functional as F
import cv2
import math
import torch
import random
from dataset.sample import SampleGenerator
import json
from dataset.visualize import KeyPointVisualizer, BBoxVisualizer
import numpy as np



class ImageTransform:
    def __init__(self, color="rgb", save="", max_rot=40):
        self.color = color
        self.prob = 0.5
        self.save = save
        self.max_rotation = max_rot
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

    def init_with_cfg(self, data_cfg):
        with open(data_cfg, "r") as load_f:
            load_dict = json.load(load_f)
        self.input_height = load_dict["input_height"]
        self.input_width = load_dict["input_width"]
        self.output_height = load_dict["output_height"]
        self.output_width = load_dict["output_width"]
        self.sigma = load_dict["sigma"]
        self.rotate = load_dict["rotate"]
        self.flip_prob = load_dict["flip_prob"]
        self.scale_factor = load_dict["scale"]
        self.kps = load_dict["kps"]
        self.SAMPLE = SampleGenerator(self.output_height, self.output_width, self.input_height, self.input_width,
                                      self.sigma)
        # if self.save:
        self.KPV = KeyPointVisualizer(self.kps, "coco")
        self.BBV = BBoxVisualizer()

    def load_img(self, img_path):
        img = cv2.imread(img_path)
        if self.color == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def img2tensor(self, img):
        ts = F.to_tensor(img)
        return ts

    def normalize(self, img):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def scale(self, img, bbox):
        x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
        width, height = x_max - x_min, y_max - y_min
        imgheight = img.shape[0]
        imgwidth = img.shape[1]
        x_enlarged_min = max(0, x_min - width * self.scale_factor / 2)
        y_enlarged_min = max(0, y_min - height * self.scale_factor / 2)
        x_enlarged_max = min(imgwidth - 1, x_max + width * self.scale_factor / 2)
        y_enlarged_max = min(imgheight - 1, y_max + height * self.scale_factor / 2)
        return [x_enlarged_min, y_enlarged_min, x_enlarged_max, y_enlarged_max]

    def flip(self, img, box, kps, kps_valid):
        import copy
        flipped_kps, flipped_valid = copy.deepcopy(kps), copy.deepcopy(kps_valid)
        flipped_img = cv2.flip(img, 1)
        img_width = img.shape[1]
        # box_w, box_tl, box_br = box[2] - box[0], box[0], box[2]
        new_box_tl, new_box_br = img_width - box[2] - 1, img_width - box[0] - 1
        flipped_box = [new_box_br, box[1], new_box_tl, box[3]]
        for l, r in self.flip_pairs:
            left_kp, right_kp = kps[l], kps[r]
            flipped_x_l, flipped_x_r = img_width - kps[r][0] - 1, img_width - kps[l][0] - 1
            flipped_kps[l], flipped_kps[r] = [flipped_x_l, right_kp[1]], [flipped_x_r, left_kp[1]]
            flipped_valid[l], flipped_valid[r] = kps_valid[r], kps_valid[l]
        return flipped_img, flipped_box, flipped_kps, flipped_valid

    def rotate_img(self, img, box, kps, valid):
        prob = random.random()
        degree = (prob-0.5) * 2 * self.max_rotation
        img_w, img_h = img.shape[0], img.shape[1]
        w, h = box[2], box[3]
        # center = (w / 2, h / 2)
        center_img = (img_w/2, img_h/2)
        # R = cv2.getRotationMatrix2D(center, degree, 1)
        R_img = cv2.getRotationMatrix2D(center_img, degree, 1)
        cos, sin = abs(R_img[0, 0]), abs(R_img[0, 1])    #degrree between -40 and 40
        kps = np.asarray(kps)
        kps_new = np.zeros(kps.shape, dtype=kps.dtype)
        x = kps[:, 0] - center_img[0]
        y = kps[:, 1] - center_img[1]
        kps_new[:, 0] = x * cos + y * sin + center_img[0]
        kps_new[:, 1] = x * sin + y * cos + center_img[1]
        # new_w = int(h * sin + w * cos)
        # new_h = int(h * cos + w * sin)
        new_img_w = int(img_h * sin + img_w * cos)
        new_img_h = int(img_h * cos + img_w * sin)
        new_img_size = (new_img_w, new_img_h)
        R_img[0, 2] += new_img_w/2 - center_img[0]
        R_img[1, 2] += new_img_h/2 - center_img[1]
        # new_size = (new_w, new_h)
        # R[0, 2] += new_w/2 - center[0]
        # R[1, 2] += new_h/2 - center[1]
        # box_new = cv2.warpAffine(box, R, dsize=new_size, borderMode=cv2.BORDER_CONSTANT, borderValue=None)
        img_new = cv2.warpAffine(img, R_img, dsize=new_img_size)
        return img_new, kps_new, valid

    def tensor2img(self, ts):
        img = np.asarray(F.to_pil_image(ts))
        return img

    def process(self, img_path, box, kps):
        raw_img = self.load_img(img_path)
        enlarged_box = self.scale(raw_img, box)
        img, pad_size, labels = self.SAMPLE.process(raw_img, enlarged_box, kps)
        inputs = self.normalize(self.img2tensor(img))
        if self.save:
            import os
            import copy
            os.makedirs("{}".format(self.save), exist_ok=True)
            cv2.imwrite("{}/raw.jpg".format(self.save), raw_img)
            cv2.imwrite("{}/cropped_padding.jpg".format(self.save), img)
            cv2.imwrite("{}/box.jpg".format(self.save), self.BBV.visualize([box], copy.deepcopy(raw_img)))
            cv2.imwrite("{}/enlarge_box.jpg".format(self.save), self.BBV.visualize([enlarged_box], copy.deepcopy(raw_img)))
            cv2.imwrite("{}/kps.jpg".format(self.save), self.KPV.visualize(copy.deepcopy(raw_img), [kps]))
            for idx in range(self.kps):
                hm = self.SAMPLE.save_hm(img, labels[idx])
                cv2.imwrite("{}/kps_{}.jpg".format(self.save, idx), cv2.resize(hm, (640, 640)))
        return inputs, labels, enlarged_box, pad_size


if __name__ == '__main__':
    # data_cfg = "../config/data_cfg/data_default.json"
    # img_path = 'sample.jpg'
    # box = [166.921, 85.08000000000001, 304.42900000000003, 479]
    # kps = [[0, 0], [0, 0], [252, 156], [0, 0], [248, 153], [198, 193], [243, 196], [182, 245], [244, 263], [0, 0],
    #        [276, 285], [197, 298], [228, 297], [208, 398], [266, 399], [205, 475], [215, 453]]
    # valid = [0, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2]
    #
    # IT = ImageTransform()
    # IT.init_with_cfg(data_cfg)
    # img = IT.load_img(img_path)
    # f_img, f_box, f_kps, f_valid = IT.flip(img, box, kps, valid)
    #
    # IT.BBV.visualize([f_box], f_img)
    # IT.KPV.visualize(f_img, [f_kps])
    # IT.BBV.visualize([box], img)
    # IT.KPV.visualize(img, [kps])
    #
    # cv2.imshow("raw", img)
    # cv2.imshow("flipped", f_img)
    # cv2.waitKey(0)

    data_cfg = "../config/data_cfg/data_default.json"
    img_path = 'sample.jpg'
    box = [166.921, 85.08000000000001, 304.42900000000003, 479]
    kps = [[0, 0], [0, 0], [252, 156], [0, 0], [248, 153], [198, 193], [243, 196], [182, 245], [244, 263], [0, 0],
           [276, 285], [197, 298], [228, 297], [208, 398], [266, 399], [205, 475], [215, 453]]
    valid = [0, 0, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2]

    max_rotate = 40
    IT = ImageTransform(max_rot=max_rotate)
    IT.init_with_cfg(data_cfg)
    img = IT.load_img(img_path)
    f_img, f_kps, f_valid = IT.rotate_img(img, box, kps, valid)

    IT.KPV.visualize(f_img, [f_kps])
    IT.KPV.visualize(img, [kps])

    cv2.imshow("raw", img)
    cv2.imshow("rotated", f_img)
    cv2.waitKey(0)

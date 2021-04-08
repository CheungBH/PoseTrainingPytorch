from torchvision.transforms import functional as F
import cv2
import math
import torch
import random
from .sample import SampleGenerator


class ImageTransform:
    def __init__(self, color="rgb"):
        self.color = color
        self.prob = 0.5
        self.max_rotation = 40
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def init_with_cfg(self, data_cfg):
        with open(data_cfg, "r") as load_f:
            load_dict = data_cfg.load(load_f)
        self.input_height = load_dict["input_height"]
        self.input_width = load_dict["input_width"]
        self.output_height = load_dict["output_height"]
        self.output_width = load_dict["output_width"]
        self.sigma = load_dict["sigma"]
        self.rotate = load_dict["rotate"]
        self.flip_prob = load_dict["flip_prob"]
        self.scale = load_dict["scale"]
        self.kps = load_dict["kps"]
        self.SAMPLE = SampleGenerator(self.output_height, self.output_width, self.input_height, self.input_width,
                                      self.sigma)

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
        left, top, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        imgheight = img.shape[1]
        imgwidth = img.shape[2]
        x = max(0, left - width * self.scale / 2)
        y = max(0, top - height * self.scale / 2)
        bottomRightx = min(imgwidth - 1, left + width * (1+self.scale / 2))
        bottomRighty = min(imgheight - 1, top + height * (1+self.scale / 2))
        return [x, y, bottomRightx, bottomRighty]

    def flip(self, img, box, kps):
        prob = random.random()
        right = [2, 4, 6, 8, 10, 12, 14, 16]
        left = [1, 3, 5, 7, 9, 11, 13, 15]
        do_flip = prob <= self.prob
        if not do_flip:
            return img, box, kps
        img = cv2.flip(img)
        for r, l in zip(right, left):
            kps[r], kps[l] = kps[l], kps[r]
        return img, box, kps

    def rotate(self, img, box, kps):
        prob = random.random()
        degree = (prob-0.5) * 2 * self.max_rotation  #degrree between -40 and 40
        radian = degree/180.0 * math.pi
        w, h = box[2], box[3]
        center = (w / 2, h / 2)
        radian_sin = math.sin(radian)
        radian_cos = math.cos(radian)
        kps_new = torch.zeros(kps.shape, dtype=kps.dtype)
        x = kps[:, 0] - center[0]
        y = kps[:, 1] - center[1]
        kps_new[:, 0] = x * radian_cos - y * radian_sin + 0.5 * center[0]
        kps_new[:, 1] = x * radian_sin + y * radian_cos + 0.5 * center[1]

        R = cv2.getRotationMatrix2D(center, degree, 1)
        cos, sin = abs(R[0, 0]), abs(R[0, 1])
        new_w = h * sin + w * cos
        new_h = h * cos + w * sin
        new_size = (new_w, new_h)
        R[0, 2] += new_size[0]/2 - center[0]
        R[1, 2] += new_size[1]/2 - center[1]
        img_new = cv2.warpAffine(img, R, dsize=new_size, borderMode=cv2.BORDER_CONSTANT, borderValue=None)
        return img_new, kps_new

    def tensor2img(self, ts):
        img = F.to_pil_image(ts)
        return img

    def process(self, img_path, box, kps):
        raw_img = self.load_img(img_path)
        enlarged_box = self.scale(raw_img, box)
        img, labels = self.SAMPLE.process(raw_img, enlarged_box, kps)
        inputs = self.img2tensor(self.normalize(img))
        return inputs, labels, enlarged_box






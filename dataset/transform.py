from torchvision.transforms import functional as F
import cv2
import random
from dataset.sample import SampleGenerator
import json
from dataset.visualize import KeyPointVisualizer, BBoxVisualizer
import numpy as np
from dataset.rotate import cv_rotate
import math
from PIL import Image


class ImageTransform:
    def __init__(self, color="rgb", save=""):
        self.color = color
        self.prob = 0.5
        self.save = save
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
        self.not_flip_idx = [0]

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
        self.rotate_prob = load_dict["rotate_prob"]
        self.SAMPLE = SampleGenerator(self.output_height, self.output_width, self.input_height, self.input_width,
                                      self.sigma)
        # if self.save:
        self.KPV = KeyPointVisualizer(self.kps, "aic")
        self.BBV = BBoxVisualizer()
        self.update_flip_pairs()

    def update_flip_pairs(self):
        if self.kps == 13:
            self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
            self.not_flip_idx = [0]
        elif self.kps == 16:
            self.flip_pairs = ([0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13])
            self.not_flip_idx = [6, 7, 8, 9]
        elif self.kps == 14:
            self.flip_pairs = ([0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11])
            self.not_flip_idx = [12, 13]
        elif self.kps == 17:
            return
        else:
            raise NotImplementedError

    def load_img(self, img_path):
        try:
            img = cv2.imread(img_path)
            if self.color == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            import sys
            print(img_path)
            sys.exit()
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
        flipped_box = [new_box_tl, box[1], new_box_br, box[3]]
        for idx in self.not_flip_idx:
            flipped_kps[idx] = [img_width - kps[idx][0] - 1, kps[idx][1]]
        for l, r in self.flip_pairs:
            left_kp, right_kp = kps[l], kps[r]
            flipped_x_l, flipped_x_r = img_width - kps[r][0] - 1, img_width - kps[l][0] - 1
            flipped_kps[l], flipped_kps[r] = [flipped_x_l, right_kp[1]], [flipped_x_r, left_kp[1]]
            flipped_valid[l], flipped_valid[r] = kps_valid[r], kps_valid[l]
        return flipped_img, flipped_box, flipped_kps, flipped_valid

    def rotate_img(self, img, box, kps, valid):
        kps_new = []
        img_h, img_w = img.shape[0], img.shape[1]
        center_img = (img_w/2, img_h/2)
        Pi_angle = -degree * math.pi / 180.0
        R_img = cv2.getRotationMatrix2D(center_img, degree, 1)
        cos, sin = abs(R_img[0, 0]), abs(R_img[0, 1])
        new_img_w = int(img_w * cos + img_h * sin)
        new_img_h = int(img_w * sin + img_h * cos)
        new_img_size = (new_img_w, new_img_h)
        R_img[0, 2] += new_img_w / 2 - center_img[0]
        R_img[1, 2] += new_img_h / 2 - center_img[1]
        img_new = cv2.warpAffine(img, R_img, dsize=new_img_size,borderMode=cv2.BORDER_CONSTANT)
        bbox_new = self.bbox_rotate(box,R_img)
        for keypoint in kps:
            kps_new.append(self.rotate_point(keypoint,R_img))
        return img_new, kps_new, valid, bbox_new


    def rotate_point(self,point,R):
        return [R[0, 0] * point[0] + R[0, 1] * point[1] + R[0, 2],
                R[1, 0] * point[0] + R[1, 1] * point[1] + R[1, 2]]

    def bbox_rotate(self,bbox,rot_mat):
        assert len(bbox) == 4
        bbox = [[bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]]]
        rot_bboxes = list()
        point1 = np.dot(rot_mat, np.array([bbox[0][0], bbox[0][1], 1]).astype(np.int32))
        point2 = np.dot(rot_mat, np.array([bbox[1][0], bbox[1][1], 1]).astype(np.int32))
        point3 = np.dot(rot_mat, np.array([bbox[2][0], bbox[2][1], 1]).astype(np.int32))
        point4 = np.dot(rot_mat, np.array([bbox[3][0], bbox[3][1], 1]).astype(np.int32))

        # 加入list中
        # bbox_new = [point1[0],point1[1],point4[0],point4[1]]
        rot_bboxes.append([[point1[0], point1[1]],
                           [point2[0], point2[1]],
                           [point3[0], point3[1]],
                           [point4[0], point4[1]]])

        return rot_bboxes

        # a = M[:, :2]  ##a.shape (2,2)
        # b = M[:, 2:]  ###b.shape(2,1)
        # b = np.reshape(b, newshape=(1, 2))
        # a = np.transpose(a)
        #
        # [left, up, right, down] = bbox
        # corner_point = np.array([[left, up], [right, up], [left, down], [right, down]])
        # corner_point = np.dot(corner_point, a) + b
        # min_left = max(int(np.min(corner_point[:, 0])), 0)
        # max_right = min(int(np.max(corner_point[:, 0])), img_shape[1])
        # min_up = max(int(np.min(corner_point[:, 1])), 0)
        # max_down = min(int(np.max(corner_point[:, 1])), img_shape[0])
        #
        # return [min_left, max_right, min_up, max_down]



    def rotate_cropped_img(self, im, kps, kps_valid, degree):
        # rotate with center
        return im, kps, kps_valid

    def tensor2img(self, ts):
        img = np.asarray(F.to_pil_image(ts))
        return img

    def process(self, img_path, box, kps, kps_valid, img_aug=False):
        raw_img = self.load_img(img_path)
        enlarged_box = self.scale(raw_img, box)
        if img_aug:
            if random.random() > 1 - self.flip_prob:
                raw_img, enlarged_box, kps, kps_valid = self.flip(raw_img, enlarged_box, kps, kps_valid)
        img, pad_size, labels = self.SAMPLE.process(raw_img, enlarged_box, kps)
        inputs = self.normalize(self.img2tensor(img))
        if img_aug:
            if random.random() > 1 - self.rotate_prob:
                prob = random.random()
                degree = (prob - 0.5) * 2 * self.rotate
                inputs = cv_rotate(inputs, degree, self.input_width, self.input_height)
                labels = cv_rotate(labels, degree, self.output_width, self.output_height)
                # raw_img, kps, kps_valid = self.rotate_cropped_img(raw_img, kps, kps_valid, degree)
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
        return inputs, labels, enlarged_box, pad_size, kps_valid

    def process_single_img(self, img_path, out_h, out_w, in_h, in_w):
        self.SAMPLE = SampleGenerator(out_h, out_w, in_h, in_w)
        img = self.load_img(img_path)
        padded_img, padded_size = self.SAMPLE.padding(img)
        # cv2.imshow("padded", padded_img)
        inputs = self.normalize(self.img2tensor(padded_img))
        return inputs, padded_size


if __name__ == '__main__':
    import copy
    data_cfg = "../config/data_cfg/data_default.json"
    img_path = 'sample.jpg'
    box = [166.921, 85.08000000000001, 304.42900000000003, 479]
    kps = [[0, 0], [0, 0], [252, 156], [0, 0], [248, 153], [198, 193], [243, 196], [182, 245], [244, 263], [0, 0],
           [276, 285], [197, 298], [228, 297], [208, 398], [266, 399], [205, 475], [215, 453]]
    valid = [[0], [0], [2], [0], [2], [2], [2], [2], [2], [0], [2], [2], [2], [2], [2], [2], [2]]

    degree = 50
    IT = ImageTransform()
    IT.init_with_cfg(data_cfg)
    img = IT.load_img(img_path)
    rot_img = copy.deepcopy(img)
    IT.KPV.visualize(img, [kps], [valid])
    IT.BBV.visualize([box], img)

    rot_img, kps, valid, _ = IT.rotate_img(rot_img, box, kps, valid)
    # f_img, f_box, f_kps, f_valid = IT.flip(img, box, kps, valid)

    # IT.BBV.visualize([f_box], f_img)
    IT.KPV.visualize(rot_img, [kps], [valid])
    IT.BBV.visualize([box], rot_img)
    cv2.imshow("raw", img)
    cv2.imshow("rot", rot_img)
    cv2.waitKey(0)

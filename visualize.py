from models.pose_model import PoseModel
from dataset.transform import ImageTransform
from dataset.draw import PredictionVisualizer
from utils.test_utils import check_option_file
import cv2
import os
import torch

posenet = PoseModel()


class ImageVisualizer:
    out_h, out_w, in_h, in_w = 64, 64, 256, 256
    def __init__(self, model_cfg, model_path, data_cfg=None, show=True):
        self.show = show
        option_file = check_option_file(model_path)
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
        posenet.load(model_path)
        self.PV = PredictionVisualizer(posenet.kps, 1, self.out_h, self.out_w, self.in_h, self.in_w, max_img=1, column=1)

    def visualize(self, img_path, save=""):
        img = cv2.imread(img_path)
        inp, padded_size = self.transform.process_single_img(img_path)
        img_meta = {
            "name": img_path,
            "enlarged_box": [0, 0, img.size(1), img.size(0)],
            "padded_size": padded_size
        }
        out = self.model(inp)
        draw = self.PV.draw_kps(out, img_meta)
        if save:
            cv2.imwrite(save, draw)
        if self.show:
            cv2.imshow("res", img)
            cv2.waitKey(0)


from models.pose_model import PoseModel
from dataset.transform import ImageTransform
from dataset.draw import PredictionVisualizer
from utils.utils import get_option_path
import cv2
import os
from utils.utils import get_corresponding_cfg
import torch

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
        self.black_id = 0

    def visualize(self, img_path, save):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save = os.path.join(save, f"{img_name}_bbg_{self.black_id}.jpg")
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
            # drawn = self.PV.draw_kps(out[0], img_meta)
            # drawn = self.PV.draw_kps_opt(out, img_meta, self.conf)
            drawn = self.PV.draw_kps_opt_black(out, img_meta, self.conf)
            if save:
                cv2.imwrite(save, drawn)
                self.black_id += 1
                print(f"{self.black_id} images saved")
            # if self.show:
            #     cv2.imshow("res", drawn)
            #     cv2.waitKey(0)



if __name__ == '__main__':
    model_path = "/home/hkuit164/Desktop/xjl/1025+1103+1121/alphapose/latest.pth"
    folder_path = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/crop_image/train/Normal"
    save_folder = "/media/hkuit164/Backup/xjl/ML_data_process/ML/0206far/crop_image/test"
    # img_path = "/media/hkuit164/Backup/ImageClassifier/data/tennis_player/train/backhand/backhand_2.jpg"
    conf = 0.05

    model_cfg = ""
    data_cfg = ""

    if not model_path or not data_cfg:
        model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])
    IV = ImageVisualizer(model_cfg, model_path, data_cfg, conf=conf)
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            IV.visualize(img_path, save_folder)

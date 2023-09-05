import os

from visualize import ImageVisualizer
from utils.utils import init_model_list


class AutoVisualizer:
    def __init__(self, src_folder, img_folder, img_dest_folder):
        self.model_ls, self.model_cfg_ls, self.data_cfg_ls, _ = init_model_list(src_folder)
        self.img_path_ls = [os.path.join(img_folder, name) for name in os.listdir(img_folder)]
        self.img_ls = [name for name in os.listdir(img_folder)]
        self.dest_folder = img_dest_folder

    def run(self):
        model_len, img_len = len(self.model_ls), len(self.img_ls)
        for model_idx, (model, model_cfg, data_cfg) in enumerate(zip(
                self.model_ls, self.model_cfg_ls, self.data_cfg_ls)):
            print("----------------[{}/{}] Processing model {}----------------".format(model_idx+1, model_len, model))
            img_dest_folder = os.path.join(self.dest_folder, model.replace("/","-")[:-4])
            IV = ImageVisualizer(model_cfg, model, data_cfg, show=False)
            os.makedirs(img_dest_folder, exist_ok=True)
            for image_idx, (image_path, image) in enumerate(zip(self.img_path_ls, self.img_ls)):
                if image_idx % 10 == 0:
                    print("[{}/{}] Processing image".format(image_idx + 1, img_len))
                IV.visualize(image_path, os.path.join(img_dest_folder, image))


if __name__ == '__main__':
    model_folder = "/home/hkuit164/Desktop/newpose/vis_folder/tennis_ball_val"
    src_img_folder = "data/tennis_ball/val"
    dest_img_folder = "vis_folder/tennis_ball/val"
    AV = AutoVisualizer(model_folder, src_img_folder, dest_img_folder)
    AV.run()


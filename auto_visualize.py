import os

from visualize import ImageVisualizer
from utils.utils import init_model_list


class AutoVisualizer:
    def __init__(self, src_folder, img_folder, img_dest_folder):
        self.model_ls, self.model_cfg_ls, self.data_cfg_ls, self.option_ls = init_model_list(src_folder)
        self.img_path_ls = [os.path.join(img_folder, name) for name in os.listdir(img_folder)]
        self.img_ls = [name for name in os.listdir(img_folder)]
        self.dest_folder = img_dest_folder

    def run(self):
        for model_idx, (model, model_cfg, data_cfg) in enumerate(zip(
                self.model_ls, self.model_cfg_ls, self.data_cfg_ls)):
            img_dest_folder = os.path.join(self.dest_folder, model[:-4])
            AutoV = ImageVisualizer(model, model_cfg, data_cfg, show=False)
            os.makedirs(img_dest_folder, exist_ok=True)
            for image_idx, (image_path, image) in enumerate(zip(self.img_path_ls, self.img_ls)):
                AutoV.visualize(image_path, os.path.join(img_dest_folder, image))



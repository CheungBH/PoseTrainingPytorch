#-*-coding:utf-8-*-
import os
from convert import Converter
from utils.utils import init_model_list


class AutoConverter:
    def __init__(self, model_folder, onnx=True, libtorch=True, onnx_sim=True):
        self.model_folder = model_folder
        self.onnx = onnx
        self.libtorch = libtorch
        self.onnx_sim = onnx_sim
        self.model_ls, self.cfg_ls, _, _ = init_model_list(self.model_folder)

    @staticmethod
    def get_superior_path(path):
        return "/".join(path.replace("\\", "/").split("/")[:-1])

    def get_onnx_path(self, model_path):
        if self.onnx:
            return os.path.join(self.get_superior_path(model_path), "model.onnx")
        else:
            return None

    def get_libtorch_path(self, model_path):
        if self.libtorch:
            return os.path.join(self.get_superior_path(model_path), "model.pt")
        else:
            return None

    def get_onnx_sim_path(self, model_path):
        if self.onnx_sim:
            return os.path.join(self.get_superior_path(model_path), "model_sim.onnx")
        else:
            return None

    def run(self):
        total_num = len(self.model_ls)
        for idx, (model, cfg) in enumerate(zip(self.model_ls, self.cfg_ls)):
            print("-------------------[{}/{}]: Begin Converting {}--------------".format(idx+1, total_num, model))
            converter = Converter(model, cfg, self.get_onnx_path(model), self.get_libtorch_path(model),
                                  self.get_onnx_sim_path(model))
            converter.convert()


if __name__ == '__main__':
    model_folder = "exp/test_init"
    convert = AutoConverter(model_folder)
    convert.run()

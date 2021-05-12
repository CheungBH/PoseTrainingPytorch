from .utils import kps_reshape, xywh2xyxy, select_kps
from .keypoint import KeyPointsProcessor


class BaseDataset:
    def __init__(self, kps):
        self.kps = kps
        self.KPP = KeyPointsProcessor()
        self.body_part_idx, self.body_part_name, self.flip_pairs = self.init_kps()

    def init_kps(self):
        return [], [], []

    def load_data(self, json_file,folder_name):
        pass
#-*-coding:utf-8-*-
import json


class KeyPointsRegister:
    def __init__(self):
        self.body_parts = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
              7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
              12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle", 17: "right ankle"}

    def init_kps(self, idx):
        if isinstance(idx, int):
            assert idx == 17 or idx == 13, "Wrong keypoints nums"
            if idx == 17:
                self.body_part_idx = [i+1 for i in range(17)]
            else:
                self.body_part_idx = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        elif isinstance(idx, list):
            pass
        else:
            raise TypeError("Your key-point register is wrong! ")
        self.body_part_name = [self.body_parts[no] for no in self.body_part_idx]

    def get_kps_name(self):
        return self.body_part_idx, self.body_part_name

    def unify_weighted(self, loss_weight):
        all_kps = [-item for item in range(len(self.body_part_idx) + 1)[1:]]
        if not loss_weight:
            return {1: all_kps}
        weighted_kps = []
        for param, kps_idx in loss_weight.items():
            weighted_kps += kps_idx
        remaining_kps = [item for item in all_kps if item not in weighted_kps]
        loss_weight[1] = remaining_kps
        return loss_weight


def parse_data_cfg(cfg):
    with open(cfg, "r") as load_f:
        load_dict = json.load(load_f)
    result = {
        "input_height": load_dict["input_height"],
        "input_width": load_dict["input_width"],
        "output_height": load_dict["output_height"],
        "output_width": load_dict["output_width"],
        "sigma": load_dict["sigma"],
        "rotate": load_dict["rotate"],
        "flip_prob": load_dict["flip_prob"],
        "scale": load_dict["scale"],
        "kps": load_dict["kps"]
    }
    return result



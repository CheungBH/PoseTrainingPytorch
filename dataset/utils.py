#-*-coding:utf-8-*-


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


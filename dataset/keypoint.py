#-*-coding:utf-8-*-
import json


class KeyPointsRegister:
    def __init__(self):
        pass

    @staticmethod
    def get_name(num):
        if num == 17:
            return ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                    'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
                    'right knee', 'left ankle', 'right ankle']
        elif num == 13:
            return ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
                    'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']


class KeyPointsProcessor:
    def __init__(self):
        self.body_parts_name = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder',
                                'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip',
                                'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']

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


if __name__ == '__main__':
    KPR = KeyPointsProcessor()
    # print([n for k, n in KPR.body_parts.items()])

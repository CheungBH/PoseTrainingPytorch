#-*-coding:utf-8-*-


class KeyPointsRegister:
    def __init__(self):
        self.base_kps = ['top_head', 'neck', 'nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder',
                         'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip',
                         'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']

    @staticmethod
    def get_name(num):
        if num == 17:
            return ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                    'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
                    'right knee', 'left ankle', 'right ankle']
        elif num == 13:
            return ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
                    'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']
        elif num == 14:
            return ['top head', 'neck', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
                    'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']
        elif num == 1:
            return ["Ball"]


class KeyPointsProcessor:
    def __init__(self):
        self.coco_parts_name = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder',
                                'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip',
                                'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']
        self.mpii_parts_name = ['top head', 'neck', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow',
                                'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee',
                                'left ankle', 'right ankle']
        self.kps13_parts_name = ['nose', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist',
                                 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle',
                                 'right ankle']
        self.ball_names = ["Ball"]

        self.coco_flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
        self.kps13_flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        self.mpii_flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.aic_flip_pairs = [[0, 3], [1, 4], [2, 5], [6, 9], [7, 10], [8, 11]]

    def init_kps(self, idx, dataset_type):
        if isinstance(idx, int):
            assert idx in [1, 13, 14, 16, 17], "Wrong keypoints nums"
            if idx == 17:
                self.body_part_idx = [i+1 for i in range(17)]
                self.body_part_name = self.coco_parts_name
                self.flip_pairs = self.coco_flip_pairs
                self.not_flip_idx = [0]
            elif idx == 14:
                self.body_part_idx = [i+1 for i in range(14)]
                self.body_part_name = self.mpii_parts_name
                self.flip_pairs = self.aic_flip_pairs
                self.not_flip_idx = [12, 13]
            elif idx == 16:
                self.body_part_idx = [i+1 for i in range(16)]
                self.body_part_name = self.mpii_parts_name
                self.flip_pairs = self.mpii_flip_pairs
                self.not_flip_idx = [6, 7, 8, 9]
            elif idx == 13:
                self.body_part_name = self.kps13_parts_name
                self.flip_pairs = self.kps13_flip_pairs
                self.not_flip_idx = [0]
                if dataset_type == "mpii" or dataset_type == "aic":
                    self.body_part_idx = [i+1 for i in range(13)]
                else:
                    self.body_part_idx = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
            elif idx == 1:
                self.body_part_idx = [1]
                self.body_part_name = []
                self.flip_pairs = []
                self.not_flip_idx = [0]
        elif isinstance(idx, list):
            raise NotImplementedError
        else:
            raise TypeError("Your key-point register is wrong! ")

    def get_kps_info(self):
        return self.body_part_idx, self.body_part_name, self.flip_pairs, self.not_flip_idx

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

#-*-coding:utf-8-*-

import torch
import cv2
import numpy as np

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

coco_p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
coco_line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                   (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                   (77, 222, 255), (255, 156, 127),
                   (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

mpii_p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE,
                BLUE]
mpii_line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]


class KeyPointVisualizer:
    def __init__(self, kps, dataset):
        self.kps = kps
        self.dot_size = 4
        if kps == 13:
            self.l_pair = [
                (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                (13, 7), (13, 8), (0, 13),  # Body
                (7, 9), (8, 10), (9, 11), (10, 12)
            ]
            self.p_color = coco_p_color
            self.line_color = coco_line_color

        elif kps == 17:
            self.l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            self.p_color = coco_p_color
            self.line_color = coco_line_color

        elif kps == 1 or dataset == "ball":
            self.l_pair = []
            self.p_color = RED
            self.line_color = []
            self.dot_size = 10
        else:
            if dataset == "mpii":
                self.l_pair = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [8, 12],
                               [12, 11], [11, 10], [8, 13], [13, 14], [14, 15]]
                self.p_color = mpii_p_color
                self.line_color = mpii_line_color
            elif dataset == "aic":
                self.l_pair = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5], [8, 7], [7, 6], [6, 9], [9, 10],
                               [10, 11], [12, 13], [0, 6], [3, 9]]
                self.p_color = mpii_p_color
                self.line_color = mpii_line_color
            else:
                raise NotImplementedError("Not a suitable dataset and kps num!")

    def visualize(self, frame, kps, kps_confs=[], conf=0.05):
        kps = torch.Tensor(kps)
        if len(kps_confs) <= 0:
            kps_confs = torch.Tensor([[[1 for _ in range(kps.shape[0])] for j in range(kps.shape[1])]])
        else:
            kps_confs = torch.Tensor(kps_confs)

        if isinstance(conf, float):
            conf = torch.tensor([conf for _ in range(self.kps)])

        for idx in range(len(kps)):
            part_line = {}
            kp_preds = kps[idx]
            kp_confs = kps_confs[idx]

            if self.kps == 17:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
                kp_confs = torch.cat((kp_confs, torch.unsqueeze((kp_confs[5, :] + kp_confs[6, :]) / 2, 0)))
                conf = torch.cat((conf, torch.unsqueeze((conf[5] + conf[6]) / 2, 0)))

            elif self.kps == 13:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[1, :] + kp_preds[2, :]) / 2, 0)))
                kp_confs = torch.cat((kp_confs, torch.unsqueeze((kp_confs[1, :] + kp_confs[2, :]) / 2, 0)))
                conf = torch.cat((conf, torch.unsqueeze((conf[1] + conf[2]) / 2, 0)))

            # Draw keypoints
            for n in range(kp_preds.shape[0]):
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                cor_conf = kp_confs[n, 0]

                if cor_x == 0 or cor_y == 0 or cor_conf < conf[n]:
                    continue

                part_line[n] = (cor_x, cor_y)
                cv2.circle(frame, (cor_x, cor_y), self.dot_size, self.p_color[n], -1)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(self.l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    cv2.line(frame, start_xy, end_xy, self.line_color[i], 8)
        return frame


class BBoxVisualizer:
    def __init__(self):
        self.color = (0, 0, 255)

    def visualize(self, bboxes, img):
        for bbox in bboxes:
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.color, 4)
        return img
    def visualize1(self,bboxes,img):
        for bbox in bboxes:
            plot_im = cv2.polylines(img, np.array(bboxes).astype(np.int32), True, (255, 0, 0), 2)
        return plot_im


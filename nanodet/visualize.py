import cv2
import numpy as np


class IDVisualizer(object):
    def __init__(self):
        pass
        # self.with_bbox = with_bbox

    def plot_bbox_id(self, id2bbox, img, color=("gold", "red"), id_pos="up", with_bbox=False):
        for idx, box in id2bbox.items():

            [x1, y1, x2, y2] = box
            if id_pos == "up":
                cv2.putText(img, "id{}".format(idx), (int((x1 + x2)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 4,
                            (255, 0, 0), 4)
            else:
                cv2.putText(img, "id{}".format(idx), (int((x1 + x2)/2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 4,
                            (255, 0, 0), 4)
            if with_bbox:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)

    # def plot_skeleton_id(self, id2ske, img):
    #     for idx, kps in id2ske.items():
    #
    #         x = np.mean(np.array([item[0] for item in kps]))
    #         y = np.mean(np.array([item[1] for item in kps]))
    #         cv2.putText(img, "id{}".format(idx), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,1,
    #                     colors["yellow"], 2)

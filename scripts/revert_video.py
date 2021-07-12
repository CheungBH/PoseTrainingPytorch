#-*-coding:utf-8-*-
import cv2
import numpy as np


def revert(video_src):
    video_dest = video_src[:-4] + "_processed.mp4"

    cap = cv2.VideoCapture(video_src)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    out = cv2.VideoWriter(video_dest, cv2.VideoWriter_fourcc(*'XVID'), 12, (height, width))

    while True:
        ret, frame = cap.read()
        if ret:
            out.write(np.rot90(np.rot90(np.rot90(frame))))
        else:
            break


if __name__ == '__main__':
    video_path = "../demo/video/passive lateral trunk elongation.mp4"
    revert(video_path)

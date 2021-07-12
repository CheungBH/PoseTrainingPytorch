from models.pose_model import PoseModel
from dataset.transform import ImageTransform
from dataset.draw import PredictionVisualizer
from utils.utils import get_option_path
import cv2
import os
from utils.utils import get_corresponding_cfg
import torch

posenet = PoseModel()


class VideoVisualizer:
    out_h, out_w, in_h, in_w = 64, 64, 256, 256

    def __init__(self, model_cfg, model_path, data_cfg=None, show=True, device="cuda"):
        self.show = show
        self.device = device
        option_file = get_option_path(model_path)
        self.transform = ImageTransform()

        if os.path.exists(option_file):
            option = torch.load(option_file)
            self.out_h, self.out_w, self.in_h, self.in_w = \
                option.output_height, option.output_width, option.input_height, option.input_width
        else:
            if data_cfg:
                self.transform.init_with_cfg(data_cfg)
                self.out_h, self.out_w, self.in_h, self.in_w = \
                    self.transform.output_height, self.transform.output_width, self.transform.input_height,self.transform.input_width
            else:
                pass

        posenet.build(model_cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        self.model.eval()
        posenet.load(model_path)
        self.PV = PredictionVisualizer(posenet.kps, 1, self.out_h, self.out_w, self.in_h, self.in_w, max_img=1, column=1)

    def visualize(self, video_path, save="", wait_key=1):
        import copy
        cap = cv2.VideoCapture(video_path)
        height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if save:
            out = cv2.VideoWriter(video_path[:-4] + "_processed.avi", cv2.VideoWriter_fourcc(*'XVID'), 12,
                                  (height, width))
        while True:
            ret, frame = cap.read()
            if ret:
                inp, padded_size = self.transform.process_frame(copy.deepcopy(frame), self.out_h, self.out_w, self.in_h, self.in_w)
                img_meta = {
                    "name": frame,
                    "enlarged_box": [0, 0, frame.shape[1], frame.shape[0]],
                    "padded_size": padded_size
                }
                if self.device != "cpu":
                    inp = inp.cuda()
                out = self.model(inp.unsqueeze(dim=0))
                drawn = self.PV.draw_kps_opt(out, img_meta)

                if save:
                    out.write(frame)
                if self.show:
                    cv2.imshow("output", drawn)
                    cv2.waitKey(wait_key)

            else:
                if save:
                    out.release()
                cap.release()
                break


if __name__ == '__main__':
    model_path = "exp/test_kps/aic_13/latest.pth"
    video_path = "demo/video/passive lateral trunk elongation_processed.mp4"

    model_cfg = ""
    data_cfg = ""

    if not model_path or not data_cfg:
        model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])
    vv = VideoVisualizer(model_cfg, model_path, data_cfg)
    vv.visualize(video_path)

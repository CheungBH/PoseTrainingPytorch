from models.pose_model import PoseModel
from dataset.transform import ImageTransform
from dataset.predict import HeatmapPredictor
from utils.utils import get_option_path
import cv2
import os
from utils.utils import get_corresponding_cfg
import torch
import numpy as np

posenet = PoseModel()


class BoxesImageVisualizer:
    out_h, out_w, in_h, in_w = 64, 64, 256, 256

    def __init__(self, model_cfg, model_path, data_cfg, show=True, device="cuda"):
        self.show = show
        self.device = device
        option_file = get_option_path(model_path)
        self.transform = ImageTransform()
        self.transform.init_with_cfg(data_cfg)

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
        self.HP = HeatmapPredictor(self.out_h, self.out_w, self.in_h, self.in_w)

    def process(self, img, boxes, batch=8):
        num_batches = len(boxes)//batch
        left_over = len(boxes) % batch
        inputs, img_metas = self.preprocess(img, boxes)

        if self.device != "cpu":
            inputs = inputs.cuda()

        outputs = []
        for num_batch in range(num_batches):
            outputs.append(self.model(inputs[num_batch*batch:(num_batch+1)*batch]))
        outputs.append(self.model(inputs[-left_over:]))
        hms = torch.cat(outputs).cpu().data
        kps, scores = self.HP.decode_hms(hms, img_metas)
        return kps, scores

    def preprocess(self, img, boxes):
        enlarged_boxes = [self.transform.scale(img, box) for box in boxes]
        img_metas = []
        inputs = []
        for box in enlarged_boxes:
            cropped_img = self.transform.SAMPLE.crop(box, img)
            inp, padded_size = self.transform.process_frame(cropped_img, self.out_h, self.out_w, self.in_h, self.in_w)
            inputs.append(inp.tolist())
            img_metas.append({
                "name": cropped_img,
                "enlarged_box": [box[0], box[1], box[2], box[3]],
                "padded_size": padded_size
            })
        return torch.tensor(inputs), img_metas

    @staticmethod
    def crop(box, img):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = img.shape[1] if x2 > img.shape[1] else x2
        y2 = img.shape[0] if y2 > img.shape[0] else y2
        cropped_img = np.asarray(img[y1: y2, x1: x2])
        return cropped_img


if __name__ == '__main__':
    model_path = "/home/hkuit164/Downloads/pytorch_model_samples/mob3/pytorch/3_best_acc.pth"

    model_cfg = ""
    data_cfg = ""

    if not model_path or not data_cfg:
        model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])

    from nanodet.nanodet import NanoDetector
    from nanodet.util import Logger, cfg, load_config
    from dataset.visualize import KeyPointVisualizer
    from nanodet.sort import Sort
    from nanodet.visualize import IDVisualizer

    cfg_file = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/nanodet-coco.yml"
    load_config(cfg, cfg_file)
    model = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/model_last.pth"

    tracker = Sort()

    video_path = "/home/hkuit164/Downloads/pexels-4.mp4"

    cap = cv2.VideoCapture(video_path)
    IDV = IDVisualizer()

    while True:
        ret, frame = cap.read()

        logger = Logger(0, use_tensorboard=False)
        predictor = NanoDetector(cfg, model, logger)
        meta, res = predictor.inference(frame)

        boxes = [box for box in res[0][0] if box[-1] > 0.35]
        id2box_tmp = tracker.update(torch.tensor([box + [1,0] for box in boxes]))
        id2box = {int(box[4]): box[:4] for box in id2box_tmp}
        BIV = BoxesImageVisualizer(model_cfg, model_path, data_cfg)
        kps, scores = BIV.process(frame, boxes)
        KPV = KeyPointVisualizer(BIV.kps, "coco")
        IDV.plot_bbox_id(id2box, frame, with_bbox=True)
        frame = KPV.visualize(frame, kps, scores)
        cv2.imshow("result", cv2.resize(frame, (1080, 720)))
        cv2.waitKey(1)



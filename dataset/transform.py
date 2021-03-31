from torchvision.transforms import functional as F
import cv2


class ImageTransform:
    def __init__(self, color="rgb"):
        self.color = color
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def init_with_cfg(self, data_cfg):
        with open(data_cfg, "r") as load_f:
            load_dict = data_cfg.load(load_f)
        self.input_height = load_dict["input_height"]
        self.input_width = load_dict["input_width"]
        self.output_height = load_dict["output_height"]
        self.output_width = load_dict["output_width"]
        self.sigma = load_dict["sigma"]
        self.rotate = load_dict["rotate"]
        self.flip_prob = load_dict["flip_prob"]
        self.scale = load_dict["scale"]
        self.kps = load_dict["kps"]

    def load_img(self, img_path):
        img = cv2.imread(img_path)
        if self.color == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def img2tensor(self, img):
        ts = F.to_tensor(img)
        return ts

    def normalize(self, img):
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def scale(self, im, bbox, rate):
        left, top, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        imght = im.shape[1]
        imgwidth = im.shape[2]
        x = max(0, left - width * rate / 2)
        y = max(0, top - height * rate / 2)
        bottomRightx = min(imgwidth - 1, left + width * (1+rate / 2))
        bottomRighty = min(imght - 1, top + height * (1+rate / 2))
        return [x, y, bottomRightx, bottomRighty]

    def flip(self, img, box, kps):
        return img, box, kps

    def rotate(self, img, kps, rot):
        return img, kps

    def tensor2img(self, ts):
        img = F.to_pil_image(ts)
        return img

    def process(self):
        pass








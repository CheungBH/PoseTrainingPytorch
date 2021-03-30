from torchvision.transforms import functional as F
import cv2


class ImageTransform:
    def __init__(self, color="rgb"):
        self.color = color
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

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

    def scale(self):
        pass

    def flip(self, img, kps):
        pass

    def rotate(self, img, kps):
        pass

    def tensor2img(self, ts):
        img = F.to_pil_image(ts)
        return img










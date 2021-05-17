import numpy as np
import torch
import cv2


class SampleGenerator:
    def __init__(self, out_height, out_width, inp_height, inp_width, gaussian=1, dist=3):
        self.out_height = out_height
        self.out_width = out_width
        self.inp_height = inp_height
        self.inp_width = inp_width
        self.gaussian = gaussian
        self.dist = dist

    def draw_gaussian(self, pt):
        hm_img = np.zeros((self.out_height, self.out_width))
        ul = [int(pt[0] - self.dist), int(pt[1] - self.dist)]
        br = [int(pt[0] + self.dist + 1), int(pt[1] + self.dist + 1)]
        size = 2 * self.dist + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        sigma = size / 4.0
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], self.out_width) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], self.out_height) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], self.out_width)
        img_y = max(0, ul[1]), min(br[1], self.out_height)
        try:
            hm_img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        except:
            a = 1
        return torch.from_numpy(hm_img)

    def locate_kp(self, ul, br, pt):
        center = torch.zeros(2)
        center[0] = (br[0] - 1 - ul[0]) / 2
        center[1] = (br[1] - 1 - ul[1]) / 2

        lenH = max(br[1] - ul[1], (br[0] - ul[0]) * self.inp_height / self.inp_width)
        lenW = lenH * self.inp_width / self.inp_height

        _pt = torch.zeros(2)
        _pt[0] = pt[0] - ul[0]
        _pt[1] = pt[1] - ul[1]
        # Move to center
        _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
        _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
        pt = (_pt * self.out_height) / lenH
        pt[0] = round(float(pt[0]))
        pt[1] = round(float(pt[1]))
        return pt.int()

    def crop(self, box, img):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = img.shape[1] if x2 > img.shape[1] else x2
        y2 = img.shape[0] if y2 > img.shape[0] else y2
        cropped_img = np.asarray(img[y1: y2, x1: x2])
        return cropped_img

    def padding(self, img):
        img_w, img_h = img.shape[1], img.shape[0]
        new_w = int(img_w * min(self.inp_width / img_w, self.inp_height / img_h))
        new_h = int(img_h * min(self.inp_width / img_w, self.inp_height / img_h))
        pad_size = [(self.inp_width - new_w)/2, (self.inp_height - new_h)/2]
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((self.inp_height, self.inp_width, 3), 128, dtype="uint8")
        canvas[(self.inp_height - new_h) // 2:(self.inp_height - new_h) // 2 + new_h,
            (self.inp_width - new_w) // 2:(self.inp_width - new_w) // 2 + new_w, :] = resized_image
        return canvas, pad_size

    @staticmethod
    def get_padded_location(padded_size, box):
        x_pad, y_pad = padded_size[1], padded_size[0]
        x_min, x_max, y_min, y_max = box[0] - x_pad//2, box[2] + x_pad//2,  box[1] - y_pad//2, box[3] + y_pad//2
        return [x_min, y_min, x_max, y_max]

    def process(self, img, enlarged_box, kps):
        cropped_im = self.crop(enlarged_box, img)
        padded_im, padded_size = self.padding(cropped_im)
        out = torch.zeros(len(kps), self.out_height, self.out_width)
        for i in range(len(kps)):
            padded_box = self.get_padded_location(padded_size, enlarged_box)
            up_left, bottom_right = padded_box[:2], padded_box[2:]
            if kps[i][0] > 0 and kps[i][0] > enlarged_box[0] and kps[i][1] > enlarged_box[1] and kps[i][0] < \
                    enlarged_box[2] and kps[i][1] < enlarged_box[3]:
                hm_part = self.locate_kp(up_left, bottom_right, kps[i])
                out[i] = self.draw_gaussian(hm_part)
        return padded_im, padded_size, out

    def save_hm(self, img, hm):
        save_img = cv2.resize(img, (self.out_height, self.out_width))
        hm_single = np.expand_dims(np.array(hm * 255, dtype="uint8"), axis=2)
        save_hm = np.concatenate((hm_single, hm_single, hm_single), axis=2)
        dst = cv2.addWeighted(save_img, 0.1, save_hm, 0.9, 0)
        return dst


if __name__ == '__main__':
    SG = SampleGenerator(80, 64, 320, 256, 1)
    loc = SG.locate_kp((100, 200), (300, 500), (150, 250))
    print(loc)
    hm = SG.draw_gaussian(loc)
    im_path = "../trash/675px-Poster-sized_portrait_of_Barack_Obama.jpg"
    im = cv2.imread(im_path)
    padded_img, pad_size = SG.padding(im)
    test_box = [100, 200, 300, 400]
    cropped_img = SG.crop(test_box, im)
    cv2.imshow("padded", padded_img)
    cv2.imshow("cropped", cropped_img)
    cv2.waitKey(0)

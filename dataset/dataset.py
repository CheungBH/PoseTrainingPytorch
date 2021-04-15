from dataset.transform import ImageTransform
import json
import torch.utils.data as data
import os
from dataset.utils import xywh2xyxy, kps_reshape

trans = list(zip(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ))


class BaseDataset(data.Dataset):
    def __init__(self, data_info, data_cfg, save=False, train=True):
        self.is_train = train
        if self.is_train:
            self.annot, self.imgs = "train_annot", "train_imgs"
        else:
            self.annot, self.imgs = "valid_annot", "valid_imgs"
        self.transform = ImageTransform(save=save)
        self.transform.init_with_cfg(data_cfg)
        self.load_data(data_info)

    def load_data(self, data_info):
        self.images, self.keypoints, self.boxes, self.ids, self.kps_valid = [], [], [], [], []
        for d in data_info:
            annotation_file = os.path.join(d["root"], d[self.annot])
            imgs, kps, boxes, ids, valid = self.load_json(annotation_file, os.path.join(d["root"], d[self.imgs]))
            self.images += imgs
            self.keypoints += kps
            self.boxes += boxes
            self.ids += ids
            self.kps_valid += valid
        # invalid_samples = [idx for idx, kp in enumerate(self.kps_valid) if sum(kp) == 0]
        # invalid_samples.sort(reverse=True)
        # for sample in invalid_samples

    def load_json(self, json_file, folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        images_res = []
        kps_valid = []
        for i in range(len(anno['images'])):
            images_res.append(anno['images'][i]['file_name'])
        for img_info in anno['annotations']:
            kp, kp_valid = kps_reshape(img_info["keypoints"])
            if not sum(kp_valid):
                continue
            # images.append(img_info['image_id'])
            images.append(os.path.join(folder_name, str(img_info['image_id']).zfill(12) + ".jpg"))
            # kps_tmp = img_info["keypoints"]
            # keypoint.append(img_info["keypoints"])
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            # xs = img_info['keypoints'][0::3]
            # ys = img_info['keypoints'][1::3]
            ids.append(img_info["id"])
            bbox.append(xywh2xyxy(img_info['bbox']))
        return images, keypoint, bbox, ids, kps_valid

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, kps, box, i, valid = \
            self.images[idx], self.keypoints[idx], self.boxes[idx], self.ids[idx], self.kps_valid[idx]
        inp, out, enlarged_box, pad_size = self.transform.process(path, box, kps)
        img_meta = {"name": path, "kps": kps, "box": box, "id": i, "enlarged_box": enlarged_box,
                    "padded_size": pad_size, "valid": valid}
        return inp, out, img_meta


if __name__ == '__main__':
    dataset = BaseDataset([["/media/hkuit155/Elements/coco/annotations/person_keypoints_train2017.json",
                          "/media/hkuit155/Elements/coco/train2017"]],"data_default.json")
    for i in range(len(dataset)):
        try:
            result = dataset[i]
        except:
            print(i)


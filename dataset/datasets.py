from dataset.transform import ImageTransform
import json
import torch.utils.data as data
import os
from dataset.utils import xywh2xyxy, kps_reshape
import torch

trans = list(zip(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ))
tensor = torch.Tensor


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
        images, keypoints, boxes, human_ids, kps_valid = [], [], [], [], []
        for item in data_info:
            for name, info in item.items():
                annotation_file = os.path.join(info["root"], info[self.annot])
                if name == "coco":
                    imgs, kps, box, human_id, valid = self.load_json_coco(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "mpii":
                    imgs, kps, box, human_id, valid = self.load_json_mpii(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "aic":
                    imgs, kps, box, human_id, valid = self.load_json_mpii(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "yoga":
                    imgs, kps, box, human_id, valid = self.load_json_yoga(annotation_file, os.path.join(info["root"], info[self.imgs]))
                else:
                    raise NotImplementedError
                images += imgs
                keypoints += kps
                boxes += box
                human_ids += human_id
                kps_valid += valid
        self.images = images
        self.keypoints = tensor(keypoints)
        self.boxes = tensor(boxes)
        self.human_ids = tensor(human_ids)
        self.kps_valid = tensor(kps_valid)

    def load_json_coco(self, json_file, folder_name):
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
            images.append(os.path.join(folder_name, str(img_info['image_id']).zfill(12) + ".jpg"))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            ids.append(img_info["id"])
            bbox.append(xywh2xyxy(img_info['bbox']))
        return images, keypoint, bbox, ids, kps_valid

    def load_json_aic(self):
        pass

    def load_json_mpii(self):
        pass

    def load_json_yoga(self):
        pass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, kps, box, i, valid = \
            self.images[idx], self.keypoints[idx], self.boxes[idx], self.human_ids[idx], self.kps_valid[idx]
        inp, out, enlarged_box, pad_size = self.transform.process(path, box, kps)
        img_meta = {"name": path, "kps": kps, "box": box, "id": i, "enlarged_box": enlarged_box,
                    "padded_size": pad_size, "valid": valid}
        return inp, out, img_meta


if __name__ == '__main__':
    data_info = [{"coco": {"root": "/media/hkuit155/Elements/coco",
                           "train_imgs": "train2017",
                           "valid_imgs": "val2017",
                           "train_annot": "annotations/person_keypoints_train2017.json",
                           "valid_annot": "annotations/person_keypoints_val2017.json"}}]
    sample_idx = 22

    data_cfg = "../config/data_cfg/data_default.json"
    dataset = BaseDataset(data_info, data_cfg)

    import cv2
    from dataset.visualize import BBoxVisualizer, KeyPointVisualizer
    bbv, kpv = BBoxVisualizer(), KeyPointVisualizer(17, "coco")

    result = dataset[sample_idx][-1]
    img = cv2.imread(result["name"])
    bbv.visualize([result["box"]], img)
    kpv.visualize(img, result["kps"].unsqueeze(dim=0))
    cv2.imshow("img", img)
    cv2.waitKey(0)



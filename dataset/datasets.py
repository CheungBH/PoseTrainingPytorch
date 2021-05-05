from dataset.transform import ImageTransform
import json
import torch.utils.data as data
import os
from dataset.utils import xywh2xyxy, kps_reshape
import cv2
import torch
from dataset.visualize import BBoxVisualizer, KeyPointVisualizer

tensor = torch.Tensor

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
        for item in data_info:
            for name, info in item.items():
                annotation_file = os.path.join(info["root"], info[self.annot])
                if name == "coco":
                    imgs, kps, boxes, ids, valid = self.load_json_coco(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "mpii":
                    imgs, kps, boxes, ids, valid = self.load_json_mpii(annotation_file, info["root"])
                elif name == "aic":
                    imgs, kps, boxes, ids, valid = self.load_json_aic(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "yoga":
                    imgs, kps, boxes, ids, valid = self.load_json_yoga(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "ceiling":
                    imgs, kps, boxes, ids, valid = self.load_json_ceiling(annotation_file, os.path.join(info["root"], info[self.imgs]))
                else:
                    raise NotImplementedError
                self.images += imgs
                self.keypoints += kps
                self.boxes += boxes
                self.ids += ids
                self.kps_valid += valid

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

    def load_json_aic(self,json_file,folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        kps_valid = []
        for i in range(len(anno['images'])):
            images.append(os.path.join(folder_name,str(anno['images'][i]['file_name'])))
        for img_info in anno['annotations']:
            kp, kp_valid = kps_reshape(img_info["keypoints"])
            if not sum(kp_valid):
                continue
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            ids.append(img_info["id"])
            bbox.append(xywh2xyxy(img_info['bbox']))

        return images, keypoint, bbox, ids, kps_valid

    def load_json_mpii(self,json_file,folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        kps_valid = []
        img_names = {}
        for i in range(len(anno['images'])):
            res = anno["images"][i]
            img_names[res["id"]] = os.path.join(folder_name, str(res['file_name']))
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["image_id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            if not sum(kp_valid):
                continue
            bbox.append(xywh2xyxy(entry['bbox']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
        name = list(img_names.keys())
        value = list(img_names.values())
        num = 0
        # for i in range(len(ids)):
        #     if name[num] == ids[i]:
        #         images.append(value[i])
        #         num += 1
        #     else:
        #         images.append()
        #         num -= 1
        return images, keypoint, bbox, ids, kps_valid

    def load_json_ceiling(self, json_file, folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        kps_valid = []
        # for i in range(len(anno['images'])):
        #     images.append(os.path.join(folder_name,str(anno['images'][i]['file_name'])))
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["image_id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            if not sum(kp_valid):
                continue
            if len(kp) == 16:
                # images.pop(i)
                continue
            images.append(os.path.join(folder_name, entry["image_id"]+'.jpg'))
            bbox.append(xywh2xyxy(entry['bbox']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
        return images[:-20], keypoint[:-20], bbox[:-20], ids[:-20], kps_valid[:-20]

    def load_json_yoga(self,json_file,folder_name):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        images_res = []
        kps_valid = []
        for i in range(len(anno['images'])):
            images_res.append(anno['images'][i]['file_name'])
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["image_id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            if not sum(kp_valid):
                continue
            bbox.append(xywh2xyxy(entry['bbox']))
            images.append(os.path.join(folder_name, str(entry['image_id']).zfill(12)))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
        return images, keypoint, bbox, ids, kps_valid

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, kps, box, i, valid = \
            self.images[idx], self.keypoints[idx], self.boxes[idx], self.ids[idx], self.kps_valid[idx]
        inp, out, enlarged_box, pad_size = self.transform.process(path, box, kps)
        img_meta = {"name": path, "kps": tensor(kps), "box": tensor(box), "id": i, "enlarged_box": tensor(enlarged_box),
                    "padded_size": tensor(pad_size), "valid": tensor(valid)}
        return inp, out, img_meta


if __name__ == '__main__':
    # data_info = [{"coco": {"root": "../../mmpose/data/coco",
    #                        "train_imgs": "train2017",
    #                        "valid_imgs": "val2017",
    #                        "train_annot": "annotations/person_keypoints_train2017.json",
    #                        "valid_annot": "annotations/person_keypoints_val2017.json"}}]
    # data_info = [{"mpii": {"root": "E:/data/mpii",
    #                        "train_imgs": "images",
    #                        "valid_imgs": "images",
    #                        "train_annot": "img/mpiitrain_annotonly_train.json",
    #                        "valid_annot": "img/mpiitrain_annotonly_test.json"}}]
    # data_info = [{"yoga": {"root": "../../Mobile-Pose/img",
    #                        "train_imgs": "yoga_train2",
    #                        "valid_imgs": "yoga_test",
    #                        "train_annot": "yoga_train2.json",
    #                        "valid_annot": "yoga_test.json"}}]
    # data_info = [{"aic": {"root": "E:/data/aic/ai_challenger",
    #                        "train_imgs": "train",
    #                        "valid_imgs": "valid",
    #                        "train_annot": "aic_train.json",
    #                        "valid_annot": "aic_val.json"}}]
    data_info = [{"ceiling": {"root": "../data/ceiling",
                             "train_imgs": "ceiling_train",
                             "valid_imgs": "ceiling_test",
                             "train_annot": "ceiling_train.json",
                             "valid_annot": "ceiling_test.json"}}]


    sample_idx = 1035
    data_cfg = "../config/data_cfg/data_default.json"
    dataset = BaseDataset(data_info, data_cfg, train=False)

    for i in range(sample_idx):
        import cv2
        from dataset.visualize import BBoxVisualizer, KeyPointVisualizer
        bbv = BBoxVisualizer()
        kpv = KeyPointVisualizer(17, "coco")
        result = dataset[i][-1]
        img = cv2.imread(result["name"])
        # print(result["box"])
        # print(result["kps"])
        bbv.visualize([result["box"]], img)
        kpv.visualize(img, result["kps"].unsqueeze(dim=0))
        cv2.imshow("img", cv2.resize(img, (720, 540)))
        cv2.waitKey(0)



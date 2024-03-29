from dataset.transform import ImageTransform
import json
import torch.utils.data as data
import os
from dataset.utils import xywh2xyxy, kps_reshape
import torch
import numpy as np

tensor = torch.Tensor


class BaseDataset(data.Dataset):
    def __init__(self, data_info, data_cfg, save=False, phase="train"):
        # self.is_train = train
        if phase == "train":
            self.annot, self.imgs = "train_annot", "train_imgs"
        elif phase == "valid":
            self.annot, self.imgs = "valid_annot", "valid_imgs"
        elif phase == "test":
            self.annot, self.imgs = "test_annot", "test_imgs"
        else:
            raise ValueError("Wrong phase of '{}'. Should be chosen from [train, valid, test]".format(phase))

        self.transform = ImageTransform(save=save)
        self.transform.init_with_cfg(data_cfg)
        self.load_data(data_info)

        self.out_h, self.out_w, self.in_h, self.in_w = self.transform.output_height, self.transform.output_width, \
                                                       self.transform.input_height, self.transform.input_width
        self.kps = self.transform.kps

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
        self.images = np.array(self.images)
        self.keypoints = np.array(self.keypoints)
        self.boxes = np.array(self.boxes)
        self.ids = np.array(self.ids)
        self.kps_valid = np.array(self.kps_valid)

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
        for i in range(len(anno)):
            images.append(os.path.join(folder_name,str(anno[i]['image_id'])+'.jpg'))
            kp, kp_valid = kps_reshape(anno[i]["keypoint_annotations"]["human1"])
            if not sum(kp_valid):
                continue
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            bbox.append(anno[i]['human_annotations']["human1"])
            ids.append(anno[i]['url'])
        return images, keypoint, bbox, ids, kps_valid

    def load_json_mpii(self, json_file, folder_name):
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
            ids.append(entry["id"])
            kp, kp_valid = kps_reshape(entry["keypoints"])
            if not sum(kp_valid):
                continue
            bbox.append(xywh2xyxy(entry['bbox']))
            keypoint.append(kp)
            kps_valid.append(kp_valid)
            images.append(img_names[entry["image_id"]])
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
        kps_valid = []
        for i in range(len(anno['annotations'])):
            entry = anno['annotations'][i]
            ids.append(entry["id"])
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
        # try:
        path, kps, box, i, valid = \
            self.images[idx], self.keypoints[idx], self.boxes[idx], self.ids[idx], self.kps_valid[idx]
        inp, out, enlarged_box, pad_size, valid = self.transform.process(path, box, kps, valid)
        # except:
        #     print(idx)
        #     path, kps, box, i, valid = \
        #         self.images[0], self.keypoints[0], self.boxes[0], self.ids[0], self.kps_valid[0]
        #     inp, out, enlarged_box, pad_size, valid = self.transform.process(path, box, kps, valid)
        img_meta = {"name": path, "kps": tensor(kps), "box": tensor(box), "id": i, "enlarged_box": tensor(enlarged_box),
                    "padded_size": tensor(pad_size), "valid": tensor(valid)}
        return inp, out, img_meta


if __name__ == '__main__':
    # data_info = [{"coco": {"root": "/media/hkuit155/Elements/coco",
    #                        "train_imgs": "train2017",
    #                        "valid_imgs": "val2017",
    #                        "train_annot": "annotations/person_keypoints_train2017.json",
    #                        "valid_annot": "annotations/person_keypoints_val2017.json"}}]
    # data_info = [{"mpii": {"root": "E:/data/mpii",
    #                        "train_imgs": "MPIIimages",
    #                        "valid_imgs": "MPIIimages",
    #                        "train_annot": "mpiitrain_annotonly_train.json",
    #                        "valid_annot": "mpiitrain_annotonly_test.json"}}]
    data_info = [{"yoga": {"root": "../data/yoga",
                           "train_imgs": "yoga_train2",
                           "valid_imgs": "yoga_eval",
                           "train_annot": "yoga_train2.json",
                           "valid_annot": "yoga_eval.json"}}]

    # data_info = [{"aic": {"root": "/media/hkuit155/Elements/data/aic/ai_challenger",
    #                        "train_imgs": "train",
    #                        "valid_imgs": "valid",
    #                        "train_annot": "aic_train.json",
    #                        "valid_annot": "aic_val.json"}}]
    # data_info = [{"ceiling": {"root": "../data/ceiling",
    #                          "train_imgs": "ceiling_train",
    #                          "valid_imgs": "ceiling_test",
    #                          "train_annot": "ceiling_train.json",
    #                          "valid_annot": "ceiling_test.json"}}]

    sample_idx = 13328

    data_cfg = "../config/data_cfg/data_13kps.json"
    dataset = BaseDataset(data_info, data_cfg, phase="train")
    # dataset.transform.save = True

    # for i in range(sample_idx):
    import cv2
    from dataset.visualize import BBoxVisualizer, KeyPointVisualizer
    bbv = BBoxVisualizer()
    kpv = KeyPointVisualizer(17, "coco")
    result = dataset[sample_idx][-1]
    img = cv2.imread(result["name"])
    # print(result["name"])
    # print(result["box"])
    # print(result["kps"])
    bbv.visualize([result["box"]], img)

    kpv.visualize(img, result["kps"].unsqueeze(dim=0))
    cv2.imshow("img", cv2.resize(img, (720, 540)))
    cv2.waitKey(0)



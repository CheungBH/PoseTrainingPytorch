from dataset.transform import ImageTransform
import torch.utils.data as data
import os
import torch
import numpy as np

tensor = torch.Tensor


class MixedDataset(data.Dataset):
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
        self.kps = self.transform.kps
        self.transform.init_with_cfg(data_cfg)
        self.load_data(data_info)

    def load_data(self, data_info):
        self.database = {}
        self.images, self.keypoints, self.boxes, self.ids, self.kps_valid = [], [], [], [], []
        for item in data_info:
            for name, info in item.items():
                annotation_file = os.path.join(info["root"], info[self.annot])
                if name == "coco":
                    from .database.coco import COCO
                    self.database[name] = COCO(self.kps)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "mpii":
                    from .database.mpii import MPII
                    self.database[name] = MPII(self.kps)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, info["root"])
                elif name == "aic":
                    from .database.aic import AIChallenger
                    self.database[name] = AIChallenger(self.kps)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "yoga":
                    from .database.yoga import YOGA
                    self.database[name] = YOGA(self.kps)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "ceiling":
                    from .database.yoga import YOGA
                    self.database[name] = YOGA(self.kps)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                else:
                    raise NotImplementedError
                self.images += imgs
                self.keypoints += kps
                self.boxes += boxes
                self.ids += ids
                self.kps_valid += valid
                self.transform.flip_pairs = self.database[name].flip_pairs

        self.images = np.array(self.images)
        self.keypoints = np.array(self.keypoints)
        self.boxes = np.array(self.boxes)
        self.ids = np.array(self.ids)
        self.kps_valid = np.array(self.kps_valid)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            path, kps, box, i, valid = \
                self.images[idx], self.keypoints[idx], self.boxes[idx], self.ids[idx], self.kps_valid[idx]
            inp, out, enlarged_box, pad_size, valid = self.transform.process(path, box, kps, valid)
        except:
            print(idx)
            path, kps, box, i, valid = \
                self.images[0], self.keypoints[0], self.boxes[0], self.ids[0], self.kps_valid[0]
            inp, out, enlarged_box, pad_size, valid = self.transform.process(path, box, kps, valid)
        img_meta = {"name": path, "kps": tensor(kps), "box": tensor(box), "id": i, "enlarged_box": tensor(enlarged_box),
                    "padded_size": tensor(pad_size), "valid": tensor(valid)}
        return inp, out, img_meta


if __name__ == '__main__':
    # data_info = [{"coco": {"root": "E:/coco",
    #                        "train_imgs": "train2017",
    #                        "valid_imgs": "val2017",
    #                        "train_annot": "annotations/person_keypoints_train2017.json",
    #                        "valid_annot": "annotations/person_keypoints_val2017.json"}}]
    # data_info = [{"mpii": {"root": "E:/data/mpii",
    #                        "train_imgs": "images",
    #                        "valid_imgs": "images",
    #                        "train_annot": "img/mpiitrain_annotonly_train.json",
    #                        "valid_annot": "img/mpiitrain_annotonly_test.json"}}]
    data_info = [{"yoga": {"root": "../../Mobile-Pose/img",
                           "train_imgs": "yoga_train2",
                           "valid_imgs": "yoga_test",
                           "train_annot": "yoga_train2.json",
                           "valid_annot": "yoga_test.json"}}]
    # data_info = [{"aic": {"root": "E:/data/aic/ai_challenger",
    #                        "train_imgs": "train",
    #                        "valid_imgs": "valid",
    #                        "train_annot": "aic_train.json",
    #                        "valid_annot": "aic_val.json"}}]
    # data_info = [{"ceiling": {"root": "../data/ceiling",
    #                          "train_imgs": "ceiling_train",
    #                          "valid_imgs": "ceiling_test",
    #                          "train_annot": "ceiling_train.json",
    #                          "valid_annot": "ceiling_test.json"}}]

    sample_idx = 12328

    data_cfg = "../config/data_cfg/data_default.json"
    dataset = MixedDataset(data_info, data_cfg, phase="train")
    dataset.transform.save = True

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


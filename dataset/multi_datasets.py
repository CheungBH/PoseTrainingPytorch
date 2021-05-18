from dataset.transform import ImageTransform
import torch.utils.data as data
import os
import torch
import numpy as np

tensor = torch.Tensor


class MixedDataset(data.Dataset):
    def __init__(self, data_info, data_cfg, save=False, phase="train"):
        # self.is_train = train
        self.phase = phase
        self.img_aug = False
        if phase == "train":
            self.annot, self.imgs = "train_annot", "train_imgs"
            self.img_aug = True
        elif phase == "valid":
            self.annot, self.imgs = "valid_annot", "valid_imgs"
        elif phase == "test":
            self.annot, self.imgs = "test_annot", "test_imgs"
        else:
            raise ValueError("Wrong phase of '{}'. Should be chosen from [train, valid, test]".format(phase))

        self.transform = ImageTransform(save=save)
        self.transform.init_with_cfg(data_cfg)

        self.kps = self.transform.kps
        self.out_h, self.out_w, self.in_h, self.in_w = self.transform.output_height, self.transform.output_width, \
                                                       self.transform.input_height, self.transform.input_width
        self.load_data(data_info)

    def load_data(self, data_info):
        self.database = {}
        self.images, self.keypoints, self.boxes, self.ids, self.kps_valid = [], [], [], [], []
        for item in data_info:
            for name, info in item.items():
                annotation_file = os.path.join(info["root"], info[self.annot])
                if name == "coco":
                    from dataset.database.coco import COCO
                    self.database[name] = COCO(self.kps, self.phase)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "mpii":
                    from dataset.database.mpii import MPII
                    self.database[name] = MPII(self.kps, self.phase)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, info["root"])
                elif name == "aic":
                    from dataset.database.aic import AIChallenger
                    self.database[name] = AIChallenger(self.kps, self.phase)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "yoga":
                    from dataset.database.yoga import YOGA
                    self.database[name] = YOGA(self.kps, self.phase)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                elif name == "ceiling":
                    from dataset.database.ceiling import CEILING
                    self.database[name] = CEILING(self.kps, self.phase)
                    imgs, kps, boxes, ids, valid = self.database[name].load_data(annotation_file, os.path.join(info["root"], info[self.imgs]))
                else:
                    raise NotImplementedError
                self.images += imgs
                self.keypoints += kps
                self.boxes += boxes
                self.ids += ids
                self.kps_valid += valid
                self.transform.flip_pairs = self.database[name].flip_pairs
                self.transform.not_flip_idx = self.database[name].not_flip_idx

        self.images = np.array(self.images)
        self.keypoints = np.array(self.keypoints)
        self.boxes = np.array(self.boxes)
        self.ids = np.array(self.ids)
        self.kps_valid = np.array(self.kps_valid)

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
        #     inp, out, enlarged_box, pad_size, valid = self.transform.process(path, box, kps, valid, self.img_aug)
        img_meta = {"name": path, "kps": tensor(kps), "box": tensor(box), "id": i, "enlarged_box": tensor(enlarged_box),
                    "padded_size": tensor(pad_size), "valid": tensor(valid)}
        return inp, out, img_meta


if __name__ == '__main__':
    import copy
    # data_info = [{"coco": {"root": "/media/hkuit155/Elements/coco",
    #                        "train_imgs": "train2017",
    #                        "valid_imgs": "val2017",
    #                        "train_annot": "annotations/person_keypoints_train2017.json",
    #                        "valid_annot": "annotations/person_keypoints_val2017.json"}}]
    # data_info = [{"mpii": {"root": "/media/hkuit155/Elements/data/mpii",
    #                        "train_imgs": "MPIIimages",
    #                        "valid_imgs": "MPIIimages",
    #                        "train_annot": "mpiitrain_annotonly_train.json",
    #                        "valid_annot": "mpiitrain_annotonly_test.json"}}]
    data_info = [{"yoga": {"root": "../data/yoga",
                           "train_imgs": "yoga_train2",
                           "valid_imgs": "yoga_eval",
                           "test_imgs": "yoga_test",
                           "train_annot": "yoga_train2.json",
                           "valid_annot": "yoga_eval.json",
                           "test_annot": "yoga_test.json",
                           }}]
    # data_info = [{"aic": {"root": "/media/hkuit155/Elements/data/aic",
    #                      "train_imgs": "ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902",
    #                      "valid_imgs": "ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911",
    #                      "train_annot": "ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
    #                      "valid_annot": "ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json"}}]
    # data_info = [{"ceiling": {"root": "../data/ceiling",
    #                          "train_imgs": "ceiling_train",
    #                          "valid_imgs": "ceiling_test",
    #                          "train_annot": "ceiling_train.json",
    #                          "valid_annot": "ceiling_test.json"}}]

    sample_idx = 18292

    data_cfg = "../config/data_cfg/data_13kps.json"
    dataset = MixedDataset(data_info, data_cfg, phase="train")

    # for i in range(sample_idx):
    import cv2
    from dataset.visualize import BBoxVisualizer, KeyPointVisualizer
    bbv = BBoxVisualizer()
    kpv = KeyPointVisualizer(dataset.kps, "mpii")
    result = dataset[sample_idx][-1]
    img = cv2.imread(result["name"])
    f_img, f_box, f_kps, f_valid = dataset.transform.flip(copy.deepcopy(img), copy.deepcopy(result["box"].tolist()),
                                                          copy.deepcopy(result["kps"].tolist()),
                                                          copy.deepcopy(result["valid"].tolist()))

    bbv.visualize([result["box"]], img)
    kpv.visualize(img, result["kps"].unsqueeze(dim=0))
    bbv.visualize([f_box], f_img)
    kpv.visualize(f_img, [f_kps])

    cv2.imshow("img", cv2.resize(img, (720, 540)))
    cv2.imshow("flipped", cv2.resize(f_img, (720, 540)))

    cv2.waitKey(0)



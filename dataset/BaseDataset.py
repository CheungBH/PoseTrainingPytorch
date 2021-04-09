from dataset.transform import ImageTransform
import json
import torch.utils.data as data

trans = list(zip(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    ))


class MyDataset(data.Dataset):
    def __init__(self, data_info, data_cfg):
        self.transform = ImageTransform()
        self.transform.init_with_cfg(data_cfg)
        self.load_data(data_info)

    def load_data(self, data_info):
        self.images, self.keypoints, self.boxes, self.ids = [], [], [], []
        for d in data_info:
            imgs, kps, boxes, ids = self.load_json(d[0])
            self.images += imgs
            self.keypoints += kps
            self.boxes += boxes
            self.ids += ids

    def load_json(self, json_file):
        anno = json.load(open(json_file))
        keypoint = []
        images = []
        bbox = []
        ids = []
        for img_info in anno['annotations']:
            images.append(img_info['file_name'])
            xs = img_info['keypoints'][0::3]
            ys = img_info['keypoints'][1::3]
            ids.append(img_info["id"])
            bbox.append(img_info['bbox'])
            new_kp = []
            for idx, idy in trans:
                new_kp.append(
                    (xs[idx], ys[idy])
                )
            keypoint.append(new_kp)
        return images, keypoint, bbox, ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, kps, box, i = self.images[idx], self.keypoints[idx], self.boxes[idx], self.ids[idx]
        inp, out, enlarged_box = self.transform.process(path, box, kps)
        img_meta = {"name": path, "kps": kps, "box": box, "id": i, "enlarged_box": enlarged_box}
        return inp, out, img_meta


if __name__ == '__main__':
    dataset = MyDataset([["/media/hkuit155/Elements/coco/annotations/person_keypoints_train2017.json"]],
                        "cfg.json")
    # result = dataset[2]
    a = 1

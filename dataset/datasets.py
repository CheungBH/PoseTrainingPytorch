from .transform import ImageTransform


class MyDataset:
    def __init__(self, data_info, data_cfg):
        self.transform = ImageTransform()
        self.transform.init_with_cfg(data_cfg)

    def load_data(self):
        self.images, self.keypoints, self.boxes, self.ids = [], [], [], []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, kps, box, i = self.images[idx], self.keypoints[idx], self.boxes[idx], self.ids[idx]
        img_meta = {"name": path, "kps": kps, "box": box, "id": i}
        inp, out = self.transform.process()
        return inp, out, img_meta


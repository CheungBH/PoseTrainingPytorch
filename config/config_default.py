import os
from src.opt import opt

computer = "server2"
device = "cuda:0"
backbone_ls = ["mobilenet", "seresnet101", "efficientnet-b0", "shufflenet"]
open_source_dataset = ["coco"]

body_parts = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
                7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
                12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle", 17: "right ankle"}

if opt.kps == 17:
    train_body_part = [i+1 for i in range(17)]
elif opt.kps == 13:
    train_body_part = [1,6,7,8,9,10,11,12,13,14,15,16,17]
else:
    raise ValueError("This keypoint num doesn't exist")
body_part_name = [body_parts[no] for no in train_body_part]


train_data = "coco"
train_info = {
    # "yoga": ["data/yoga/images", "data/yoga/test.h5", 2],
    # "test": ["data/data/images", "data/data/test.h5", 20]
    "coco": ["G:/MB155_data/images/images", "data/coco/annot_coco.h5", 5887],
    # "ai_challenger": ["data/ai_challenger/images", "data/ai_challenger/ai_c_anno.h5", 6000],
    # "ceiling": ["data/ceiling/surface", "data/ceiling/ceiling_cameras_686.h5", 100]
    # "../trainalpha/data/coco/images": "../trainalpha/data/coco/annot_coco.h5",
    # "data/yoga/images": "data/yoga/test.h5",
}

sparse_decay_time = 0.5
warm_up = {5: 0.01, 10: 0.1}

if opt.loss_allocate == 0:
    loss_param = {1: list(range(opt.kps))}
elif opt.loss_allocate == 1:
    loss_param = {3: [-1, -2, -7, -8, -11, -12]}
    loss_param[1] = [-item for item in list(range(opt.kps + 1))[1:] if -item not in loss_param[3]]
else:
    raise ValueError

bad_epochs = {30: 10}


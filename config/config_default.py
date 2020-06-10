import os

device = "cuda:0"
backbone_ls = ["mobilenet", "seresnet101", "efficientnet-b0", "shufflenet"]
open_source_dataset = ["coco", "ai_challenger"]

DUCs = [640, 320]
body_parts = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
                7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
                12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle", 17: "right ankle"}

# train_body_part = [1,6,7,8,9,10,11,12,13,14,15,16,17]
train_body_part = [i+1 for i in range(17)]

train_data = "coco"
train_info = {
    # "yoga": ["data/yoga/images", "data/yoga/test.h5", 2],
    # "test": ["data/data/images", "data/data/test.h5", 20]
    # "coco": ["../trainalpha/data/coco/images", "../trainalpha/data/coco/annot_coco.h5", 5887],
    # "ai_challenger": ["data/ai_challenger/images", "data/ai_challenger/ai_c_anno.h5", 6000],
    "ceiling": ["data/ceiling/surface", "data/ceiling/ceiling_cameras_686.h5", 100]
    # "../trainalpha/data/coco/images": "../trainalpha/data/coco/annot_coco.h5",
    # "data/yoga/images": "data/yoga/test.h5",
}

lr_decay = {0.7: 0.1, 0.9: 0.01}

import os

device = "cuda:0"

backbone_ls = ["mobilenet", "seresnet101", "efficientnet-b0", "shufflenet"]

backbone = "mobilenet"
seresnet_cfg = "config/pose_cfg/seresnet_cfg.txt"   # if origin, model_cfg = None
efficient_type = "b0"
mobile_setting = None
    # [[1, 14, 1, 1],
    #             [6, 24, 2, 2],
    #             [6, 28, 3, 2],
    #             [6, 48, 4, 2],
    #             [6, 72, 3, 1],
    #             [6, 120, 3, 2],
    #             [6, 318, 1, 1]]
DUCs = [640, 320]

loadModel = None
save_folder = "gray"

body_parts = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
                7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
                12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle", 17: "right ankle"}

# train_body_part = [1,6,7,8,9,10,11,12,13,14,15,16,17]
train_body_part = [i+1 for i in range(17)]
sigma = 1
hmGauss = 1

train_data = "coco"
train_info = {
    # "yoga": ["data/yoga/images", "data/yoga/test.h5", 2],
    "coco": ["../trainalpha/data/coco/images", "../trainalpha/data/coco/annot_coco.h5", 5887],
    # "ai_challenger": ["data/ai_challenger/images", "data/ai_challenger/AI_challenger_anno.h5", 300]
    # "../trainalpha/data/coco/images": "../trainalpha/data/coco/annot_coco.h5",
    # "data/yoga/images": "data/yoga/test.h5",
}

sparse = False
sparse_s = 5e-8

train_batch = 64
val_batch = 128
epochs = 500
save_interval = 1


opt_method = "adam"  # "rmsprop"
lr = 1e-3
momentum = 0
weightDecay = 0

train_mum_worker = 4
val_num_worker = 3

input_width = 256
input_height = 320
output_width = pow(2, (len(DUCs)+1)) * 8
output_height = pow(2, (len(DUCs)+1)) * 10

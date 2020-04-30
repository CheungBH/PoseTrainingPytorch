import os

device = "cuda:0"

backbone_ls = ["mobilenet", "seresnet101"]

backbone = "mobilenet"
seresnet_cfg = "config/pose_cfg/seresnet_cfg.txt"   # if origin, model_cfg = None
mobile_setting = None

loadModel = None
save_folder = "test"

body_parts = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
                7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
                12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle", 17: "right ankle"}

# train_body_part = [1,6,7,8,9,10,11,12,13,14,15,16,17]
train_body_part = [i+1 for i in range(17)]
sigma = 1

train_data = "coco"
train_data_path = "G:/images"
train_data_anno = "data/coco/annot_coco.h5"
print(os.path.isdir(train_data_path))

sparse = False
sparse_s = 1e-7

train_batch = 12
val_batch = 32
epochs = 500
save_interval = 1


opt_method = "adam"  # "rmsprop"
lr = 1e-3
momentum = 0
weightDecay = 0

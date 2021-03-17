import os
from src.opt import opt

computer = "laptop win"
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

# heatmap = {"seresnet101": [64, 80, 256, 320],
#            "seresnet18": [56, 56, 224, 224]}

# if opt.model_hm:
#     opt.outputResW, opt.outputResH, opt.inputResW, opt.inputResH = heatmap[opt.model_hm]

train_data = "coco"
if opt.dataset == "multiple":
    train_info = {
        # "yoga": ["data/yoga/images", "data/yoga/test.h5", 2],
        # "coco": ["G:/MB155_data/images/images", "data/coco/annot_coco.h5", 5887],
        # "ai_challenger": ["data/ai_challenger/images", "data/ai_challenger/ai_c_anno.h5", 1000],
        "ceiling": ["data/ceiling/ceiling_train", "data/ceiling/ceiling_train.h5", 100],
        # "underwater": ["data/0612_underwater_rename/0612_underwater_Img",
        #                "data/0612_underwater_rename/0612_underwater_all_rename.h5", 100]
    }
elif opt.dataset == "ceiling":
    train_info = {"ceiling": ["data/ceiling/ceiling_train", "data/ceiling/ceiling_train.h5", 100],}
elif opt.dataset == "underwater":
    train_info = {"underwater": ["data/0612_underwater_rename/0612_underwater_Img",
                                 "data/0612_underwater_rename/0612_underwater_all_rename.h5", 100]}
elif opt.dataset == "coco":
    train_info = {"coco": ["G:/MB155_data/images/images", "data/coco/annot_coco.h5", 5887]}
elif opt.dataset == "aic":
    train_info = {"ai_challenger": ["data/ai_challenger/images", "data/ai_challenger/ai_c_anno.h5", 1000],}
elif opt.yoga == "yoga":
    train_info = {"yoga": ["data/ai_add_searchedyoga/ai_add_searchedyoga_train",
                           "data/ai_add_searchedyoga/ai_add_searchedyoga_train.h5", 1500], }
else:
    raise ValueError("Wrong training data name!")

sparse_decay_dict = {0.5: 0.1}
warm_up = {1: 0.1, 2: 0.5}

if opt.loss_weight == 0:
    loss_weight = {}
elif opt.loss_weight == 1:
    loss_weight = {
        1.5: [-1, -2, -3, -4, -7, -8, -9, -10]
    }
elif opt.loss_weight == 2:
    loss_weight = {
        1.5: [-1, -2, -7, -8],
        1.2: [-3, -4, -9, -10],
    }
else:
    raise ValueError()

bad_epochs = {20: 0.01}
patience_decay = {1: 0.75, 2: 0.5, 3: 0.5}
lr_decay_dict = {0.7: 0.1, 0.9: 0.01}


if __name__ == '__main__':
    opt.loss_allocate = 1
    print(loss_weight)

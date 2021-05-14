from config.opt import opt

computer = "laptop win"
device = "cuda:0"

body_parts = {1: "nose", 2: "left eye", 3: "right eye", 4: "left ear", 5: "right ear", 6: "left shoulder",
              7: "right shoulder", 8: "left elbow", 9: "right elbow", 10: "left wrist", 11: "right wrist",
              12: "left hip", 13: "right hip", 14: "left knee", 15: "right knee", 16: "left ankle", 17: "right ankle"}


datasets_info = {"coco": {"root": "/media/hkuit155/Elements/coco",
                          "train_imgs": "train2017",
                          "valid_imgs": "val2017",
                          "test_imgs": "val2017",
                          "train_annot": "annotations/person_keypoints_train2017.json",
                          "valid_annot": "annotations/person_keypoints_val2017.json",
                          "test_annot": "annotations/person_keypoints_val2017.json"
                          },
                 "aic": {"root": "/media/hkuit155/Elements/data/aic",
                         "train_imgs": "ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902",
                         "valid_imgs": "ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911",
                         "test_imgs": "ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911",
                         "train_annot": "ai_challenger_keypoint_train_20170902/keypoint_train_annotations_20170909.json",
                         "valid_annot": "ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json",
                         "test_annot": "ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json",
                         },
                 "yoga": {"root": "data/ai_challenger",
                          "train_imgs": "train",
                          "valid_imgs": "val",
                          "test_imgs": "val",
                          "train_annot": "ai_challenger_train.json",
                          "valid_annot": "ai_challenger_valid.json",
                          "test_annot": "ai_challenger_valid.json",
                          },
                 "ceiling": {"root": "data/ceiling",
                             "train_imgs": "ceiling_train",
                             "valid_imgs": "ceiling_test",
                             "test_imgs": "ceiling_test",
                             "train_annot": "ceiling_train.json",
                             "valid_annot": "ceiling_test.json",
                             "test_annot": "ceiling_test.json",
                             },
                 }


if opt.dataset == "multiple":
    dataset = ["coco", "aic", "mpii"]
    train_info = [{key: item for key, item in datasets_info.items() if key in dataset}]
else:
    train_info = [{opt.dataset: datasets_info[opt.dataset]}]

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
sparse_decay_dict = {0.5: 0.1}
warm_up = {1: 0.1, 2: 0.5}

if __name__ == '__main__':
    opt.loss_allocate = 1
    print(loss_weight)

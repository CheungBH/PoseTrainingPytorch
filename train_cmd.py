#-*-coding:utf-8-*-

cmds = [
    # "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone seresnet101 --struct 1 --expFolder coco_aic --expID seresnet_origin_13kps --nEpochs 500 --loadModel exp/coco_aic/seresnet_origin_13kps/model_198.pkl --trainBatch 16 --validBatch 16",
    "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone seresnet101 --struct 0 --expFolder transfer_ceiling --expID origin --nEpochs 100 --loadModel exp/duc_se.pth --trainBatch 16 --validBatch 16",

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)

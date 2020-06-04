#-*-coding:utf-8-*-

cmds = [
    "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco1 --expID mob_origin_13kps --nEpochs 500 --loadModel exp/coco1/mob_origin_13kps/model_59.pkl --trainBatch 64 --validBatch 64",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)

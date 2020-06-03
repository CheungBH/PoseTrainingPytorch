#-*-coding:utf-8-*-

cmds = [
    "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone mobilenet --struct 0 --expFolder test --expID test --trainBatch 64 --validBatch 64",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)

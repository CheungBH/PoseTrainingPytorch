cmds = [
    "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone seresnet101 --struct 0 --expFolder transfer_ceiling --expID freeze_origin --loadModel exp/duc_se.pth --trainBatch 8 --valBatch 8"
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
cmds = [

    "CUDA_VISIBLE_DEVICES=0 python auto_trainer.py --dataset aic --expFolder seresnet --expID seresnet18_3DUC --cfg "
    "models/cfg/default/cfg_seresnet18_13kps_3DUC.json --nEpochs 120 --outputResH 160 --outputResW 128",
    "CUDA_VISIBLE_DEVICES=1 python auto_trainer.py --dataset aic --expFolder seresnet --expID seresnet50_3DUC --cfg "
    "models/cfg/default/cfg_seresnet50_13kps_3DUC.json --nEpochs 120 --outputResH 160 --outputResW 128",
    "CUDA_VISIBLE_DEVICES=2 python auto_trainer.py --dataset aic --expFolder seresnet --expID seresnet18_2DUC --cfg "
    "models/cfg/default/cfg_seresnet18_13kps.json --nEpochs 120",
    "CUDA_VISIBLE_DEVICES=3 python auto_trainer.py --dataset aic --expFolder seresnet --expID seresnet50_2DUC --cfg "
    "models/cfg/default/cfg_seresnet50_13kps.json --nEpochs 120",

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
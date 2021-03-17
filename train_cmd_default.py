cmds = [

    "python auto_trainer.py --dataset ceiling --expFolder kps_test --expID resnet18 --backbone seresnet18 "
    "--cfg models/cfg/default/cfg_resnet18_13kps.json --nEpochs 100",
    "python auto_trainer.py --dataset ceiling --expFolder kps_test --expID resnet50 --backbone seresnet50 "
    "--cfg models/cfg/default/cfg_resnet50_13kps.json --nEpochs 10",
    "python auto_trainer.py --dataset ceiling --expFolder kps_test --expID seresnet18 --backbone seresnet18 "
    "--cfg models/cfg/default/cfg_seresnet18_13kps_all_se.json --nEpochs 100",
    "python auto_trainer.py --dataset ceiling --expFolder kps_test --expID seresnet101 --backbone seresnet101 "
    "--cfg models/cfg/default/cfg_seresnet101_13kps.json --nEpochs 10",

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
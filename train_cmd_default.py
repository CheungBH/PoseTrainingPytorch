cmds = [
    "python train_opt.py --dataset coco --expFolder prune_test --expID coco_original --backbone seresnet18",
    "python train_opt.py --dataset coco --expFolder prune_test --expID coco_s1E-4 --backbone seresnet18 --loadModel ",
    "python train_opt.py --dataset coco --expFolder prune_test --expID coco_s1E-3 --backbone seresnet18 --loadModel ",
    "python train_opt.py --dataset coco --expFolder prune_test --expID coco_s5E-4 --backbone seresnet18 --loadModel ",
    "python train_opt.py --dataset aic --expFolder prune_test --expID aic_original --backbone seresnet18",
    "python train_opt.py --dataset aic --expFolder prune_test --expID aic_s1E-4 --backbone seresnet18 --loadModel ",
    "python train_opt.py --dataset aic --expFolder prune_test --expID aic_s1E-3 --backbone seresnet18 --loadModel ",
    "python train_opt.py --dataset aic --expFolder prune_test --expID aic_s5E-4 --backbone seresnet18 --loadModel ",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
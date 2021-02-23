cmds = [

    "python auto_trainer.py --dataset ceiling --expFolder auto_test_pckh --expID 2 --backbone seresnet18 --addDPG False --nEpoch 100 --se_ratio 16",
    "python auto_trainer.py --dataset ceiling --expFolder auto_test_pckh --expID 1 --backbone seresnet18 --addDPG True --nEpoch 100",
    "python auto_trainer.py --dataset ceiling --expFolder auto_test_pckh --expID 3 --backbone seresnet18 --LR 0.1 --addDPG True --nEpoch 10",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
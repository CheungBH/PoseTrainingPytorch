cmds = [

'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG True --kps 17 --validBatch 64 --trainBatch 64 --optMethod sgd --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 0 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 65',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG False --kps 17 --validBatch 64 --trainBatch 64 --optMethod sgd --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 0 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 66',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG True --kps 17 --validBatch 64 --trainBatch 64 --optMethod rmsprop --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 1.00E-04 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 67',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG False --kps 17 --validBatch 64 --trainBatch 64 --optMethod rmsprop --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 1.00E-04 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 68',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG True --kps 17 --validBatch 64 --trainBatch 64 --optMethod adam --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 1.00E-04 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 69',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG False --kps 17 --validBatch 64 --trainBatch 64 --optMethod adam --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 1.00E-04 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 70',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG True --kps 17 --validBatch 64 --trainBatch 64 --optMethod sgd --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 1.00E-04 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 71',
'CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --DUC 0 --dataset coco --addDPG False --kps 17 --validBatch 64 --trainBatch 64 --optMethod sgd --nEpochs 200 --LR 1.00E-03 --hmGauss 1 --ratio 4 --momentum 0.9 --weightDecay 1.00E-04 --save_interval 20 --expFolder alphapose-coco_mobile_17kps --expID 72',

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
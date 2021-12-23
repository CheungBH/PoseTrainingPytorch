cmds = [

    # 'CUDA_VISIBLE_DEVICES= python auto_trainer.py --backbone seresnet18 --dataset multiple --se_ratio 16 --kps 13 --input_height 256 --input_width 256 --output_height 128 --output_width 128 --validBatch 128 --trainBatch 128 --optMethod adam --freeze 0 --sparse 2.00E-07 --nEpochs 100 --LR 0.001 --sigma 1 --momentum 0 --weightDecay 0 --save_interval 20 --loadModel weights/out_128/seres18_7/latest.pth --expFolder pretrain_13kps-mixed_out128_sparse --expID 22'
    # 'CUDA_VISIBLE_DEVICES= python auto_trainer.py --backbone seresnet18 --dataset multiple --se_ratio 16 --kps 13 --input_height 256 --input_width 256 --output_height 128 --output_width 128 --validBatch 128 --trainBatch 128 --optMethod rmsprop --freeze 0 --sparse 5.00E-07 --nEpochs 100 --LR 0.001 --sigma 1 --momentum 0 --weightDecay 0 --save_interval 20 --loadModel weights/out_128/seres18_7/latest.pth --expFolder pretrain_13kps-mixed_out128_sparse --expID 23'
    # 'CUDA_VISIBLE_DEVICES= python auto_trainer.py --backbone seresnet18 --dataset multiple --se_ratio 16 --kps 13 --input_height 256 --input_width 256 --output_height 128 --output_width 128 --validBatch 128 --trainBatch 128 --optMethod rmsprop --freeze 0 --sparse 2.00E-07 --nEpochs 100 --LR 0.001 --sigma 1 --momentum 0 --weightDecay 0 --save_interval 20 --loadModel weights/out_128/seres18_7/latest.pth --expFolder pretrain_13kps-mixed_out128_sparse --expID 24'

]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
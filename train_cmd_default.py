cmds = [
    "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "sE-6_lrE-3 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 1E-3 "
    "--patience 20 --nEpochs 100 --sparse_s 1E-6 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",
    "CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "s5E-6_lrE-3 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 1E-3 "
    "--patience 20 --nEpochs 100 --sparse_s 5E-6 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",

    "CUDA_VISIBLE_DEVICES=1 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "s5E-7_lrE-3 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 1E-3 "
    "--patience 20 --nEpochs 100 --sparse_s 5E-7 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",
    "CUDA_VISIBLE_DEVICES=1 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "sE-6_lrE-4 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 1E-4 "
    "--patience 20 --nEpochs 100 --sparse_s 1E-6 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",

    "CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "sE-6_lr5E-4 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 5E-4 "
    "--patience 20 --nEpochs 100 --sparse_s 1E-6 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",
    "CUDA_VISIBLE_DEVICES=2 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "s5E-6_lr5E-4 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 5E-4 "
    "--patience 20 --nEpochs 100 --sparse_s 5E-6 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",

    "CUDA_VISIBLE_DEVICES=3 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "s5E-7_lr5E-4 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 5E-4 "
    "--patience 20 --nEpochs 100 --sparse_s 5E-7 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",
    "CUDA_VISIBLE_DEVICES=3 python train_opt.py --backbone mobilenet --struct 0 --expFolder coco_mob_sparse --expID "
    "s5E-7_lr1E-4 --loadModel pretrained/mobile_13/86/86_best_acc.pkl --trainBatch 64 --validBatch 64 --LR 1E-4 "
    "--patience 20 --nEpochs 100 --sparse_s 5E-7 --kps 13 --lr_decay_time 1 --dataset coco --optMethod adam",
]

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    os.system(cmd)
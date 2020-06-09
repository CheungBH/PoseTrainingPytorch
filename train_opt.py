# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------


# CUDA_VISIBLE_DEVICES=0 python train_opt.py --backbone mobilenet --struct 0 --expFolder test --expID test --trainBatch 32


import torch
import cv2
import torch.utils.data
from torch.autograd import Variable
import sys
import torch.nn as nn
from dataset.coco_dataset import Mscoco, MyDataset
from tqdm import tqdm
from utils.eval import DataLogger, accuracy
from utils.img import flip, shuffleLR
from src.opt import opt
from tensorboardX import SummaryWriter
import os
import config.config as config
from utils.utils import generate_cmd, adjust_lr
import argparse

from utils.compute_flops import print_model_param_flops, print_model_param_nums
from test import draw_kps, draw_hms


if opt.backbone == "mobilenet":
    from models.mobilenet.MobilePose import createModel
    from config.model_cfg import mobile_opt as model_ls
elif opt.backbone == "seresnet101":
    from models.seresnet.FastPose import createModel
    from config.model_cfg import seresnet_cfg as model_ls
elif opt.backbone == "efficientnet":
    from models.efficientnet.EfficientPose import createModel
    from config.model_cfg import efficientnet_cfg as model_ls
elif opt.backbone == "shufflenet":
    from models.shufflenet.ShufflePose import createModel
    from config.model_cfg import shufflenet_cfg as model_ls
else:
    raise ValueError("Your model name is wrong")

model_cfg = model_ls[opt.struct]
print(model_cfg)


try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False


device = config.device
opt.device = device
save_folder = opt.expID
dataset = opt.expFolder
optimize = opt.optMethod

os.makedirs("log/{}".format(dataset), exist_ok=True)

torch.backends.cudnn.benchmark = True


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)

    for i, (inps, labels, setMask, img_info) in enumerate(train_loader_desc):
        if device != "cpu":
            inps = inps.cuda().requires_grad_()
            labels = labels.cuda()
            setMask = setMask.cuda()
        else:
            inps = inps.requires_grad_()
        out = m(inps)

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if opt.sparse_s != 0:
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.weight.grad.data.add_(opt.sparse_s * torch.sign(mod.weight.data))

        optimizer.step()
        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    drawn_kp, drawn_hm = False, False
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, img_info) in enumerate(val_loader_desc):
        if device != "cpu":
            inps = inps.cuda()
            labels = labels.cuda()
            setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            if not drawn_kp:
                kps_img, have_kp = draw_kps(out, img_info)
                # if have_kp:
                drawn_kp = True
                writer.add_image("result of epoch {}".format(opt.epoch), cv2.imread("img.jpg")[:, :, ::-1],
                                 dataformats='HWC')
                # else:
                #     pass
                drawn_hm = True
                hm = draw_hms(out[0])
                writer.add_image("result of epoch {} --> heatmap".format(opt.epoch), hm)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip(inps))
            flip_out = flip(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)


        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():
    cmd_ls = sys.argv[1:]
    cmd = generate_cmd(cmd_ls)
    # Prepare Dataset

    train_dataset = MyDataset(config.train_info, train=True)
    val_dataset = MyDataset(config.train_info, train=False)
    # for k, v in config.train_info.items():
    #     pass
    # train_dataset = Mscoco(v, train=True)
    # val_dataset = Mscoco(v, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.trainNW,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=True, num_workers=opt.valNW, pin_memory=True)

    # for k, v in config.train_info.items():
    #     train_dataset = Mscoco([v[0], v[1]], train=True, val_img_num=v[2])
    #     val_dataset = Mscoco([v[0], v[1]], train=False, val_img_num=v[2])
    #
    # train_loaders[k] = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=config.train_mum_worker,
    #     pin_memory=True)
    #
    # val_loaders[k] = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=config.val_batch, shuffle=False, num_workers=config.val_num_worker, pin_memory=True)
    #
    # train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=config.train_mum_worker,
    #         pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(
    #         val_dataset, batch_size=config.val_batch, shuffle=False, num_workers=config.val_num_worker, pin_memory=True)

    # assert train_loaders != {}, "Your training data has not been specific! "

    # Model Initialize
    if device != "cpu":
        m = createModel(cfg=model_cfg).cuda()
    else:
        m = createModel(cfg=model_cfg).cpu()

    begin_epoch = 0
    pre_train_model = opt.loadModel
    flops = print_model_param_flops(m)
    print("FLOPs of current model is {}".format(flops))
    params = print_model_param_nums(m)
    print("Parameters of current model is {}".format(params))

    if pre_train_model:
        info_path = os.path.join("exp", dataset, save_folder, "option.pkl")
        info = torch.load(info_path)
        begin_epoch = int(pre_train_model.split("_")[-1][:-4]) + 1
        print('Loading Model from {}'.format(pre_train_model))
        m.load_state_dict(torch.load(pre_train_model))
        opt.trainIters = info.trainIters
        opt.valIters = info.valIters
        os.makedirs("exp/{}/{}".format(dataset, save_folder), exist_ok=True)

    else:
        print('Create new model')
        with open("log/{}/{}.txt".format(dataset, save_folder), "a+") as f:
            f.write("FLOPs of current model is {}\n".format(flops))
            f.write("Parameters of current model is {}\n".format(params))
        if not os.path.exists("exp/{}/{}".format(dataset, save_folder)):
            try:
                os.mkdir("exp/{}/{}".format(dataset, save_folder))
            except FileNotFoundError:
                os.mkdir("exp/{}".format(dataset))
                os.mkdir("exp/{}/{}".format(dataset, save_folder))
        with open("exp/{}/{}/cmd.txt".format(dataset, save_folder), "w") as f:
            f.write(cmd)

    if optimize == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=opt.LR,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weightDecay)
    elif optimize == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=opt.LR,
            weight_decay=opt.weightDecay
        )
    else:
        raise Exception

    if mix_precision:
        m, optimizer = amp.initialize(m, optimizer, opt_level="O1")

    writer = SummaryWriter(
        'tensorboard/{}/{}'.format(dataset, save_folder), comment=cmd)

    # Model Transfer
    if device != "cpu":
        m = torch.nn.DataParallel(m).cuda()
        criterion = torch.nn.MSELoss().cuda()
    else:
        m = torch.nn.DataParallel(m)
        criterion = torch.nn.MSELoss()

    # rnd_inps = Variable(torch.rand(3, 3, 224, 224), requires_grad=True)
    # writer.add_graph(m, rnd_inps)

    # Start Training
    for i in range(opt.nEpochs)[begin_epoch:]:

        opt.epoch = i

        log = open("log/{}/{}.txt".format(dataset, save_folder), "a+")
        print('############# Starting Epoch {} #############'.format(i))
        log.write('############# Starting Epoch {} #############\n'.format(i))

        for name, param in m.named_parameters():
            writer.add_histogram(
                name, param.clone().data.to("cpu").numpy(), i)

        optimizer, lr = adjust_lr(optimizer, i, config.lr_decay, opt.nEpochs)
        writer.add_scalar("lr", lr, i)
        print("epoch {}: lr {}".format(i, lr))

        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))
        log.write('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}\n'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))
        #
        opt.acc = acc
        opt.loss = loss
        m_dev = m.module

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)

        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), i)

        print('Valid:-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))
        log.write('Valid:-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}\n'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))
        log.close()

        if i % opt.save_interval == 0:
            torch.save(
                m_dev.state_dict(), 'exp/{}/{}/model_{}.pkl'.format(dataset, save_folder, i))
            torch.save(
                opt, 'exp/{}/{}/option.pkl'.format(dataset, save_folder, i))
            torch.save(
                optimizer, 'exp/{}/{}/optimizer.pkl'.format(dataset, save_folder))

    writer.close()


if __name__ == '__main__':
    main()

# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

import torch
import torch.utils.data
import torch.nn as nn
from dataset.coco_dataset import Mscoco, MyDataset
from tqdm import tqdm
from utils.eval import DataLogger, accuracy
from utils.img import flip, shuffleLR
from src.opt import opt
from tensorboardX import SummaryWriter
import os
import config.config as config
from utils.compute_flops import print_model_param_flops, print_model_param_nums
from test import draw_kps


if config.backbone == "mobilenet":
    from models.mobilenet.MobilePose import createModel
    model_cfg = config.mobile_setting
elif config.backbone == "seresnet101":
    from models.seresnet.FastPose import createModel
    model_cfg = config.seresnet_cfg
elif config.backbone == "efficientnet":
    from models.efficientnet.EfficientPose import createModel
    model_cfg = "efficientnet-" + config.efficient_type
elif config.backbone == "shufflenet":
    from models.shufflenet.ShufflePose import createModel
    model_cfg = ""
else:
    raise ValueError("Your model name is wrong")

try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False

device = config.device
save_folder = config.save_folder
dataset = config.train_data
optimize = config.opt_method

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

        if config.sparse:
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.weight.grad.data.add_(config.sparse_s * torch.sign(mod.weight.data))

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
    draw_kp = False
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

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip(inps))
            flip_out = flip(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        # if not draw_kp:
        #     draw_kp = True
        #     kps_img = draw_kps(out)

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
    # Prepare Dataset

    train_dataset = MyDataset(config.train_info, train=True)
    val_dataset = MyDataset(config.train_info, train=False)
    # for k, v in config.train_info.items():
    #     pass
    # train_dataset = Mscoco(v, train=True)
    # val_dataset = Mscoco(v, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch, shuffle=True, num_workers=config.train_mum_worker,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.val_batch, shuffle=True, num_workers=config.val_num_worker, pin_memory=True)

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
    pre_train_model = config.loadModel
    flops = print_model_param_flops(m)
    print("FLOPs of current model is {}".format(flops))
    params = print_model_param_nums(m)
    print("Parameters of current model is {}".format(params))

    if pre_train_model:
        print('Loading Model from {}'.format(pre_train_model))
        m.load_state_dict(torch.load(pre_train_model))
        opt.trainIters = config.train_batch * (begin_epoch-1)
        opt.valIters = config.val_batch * (begin_epoch-1)
        begin_epoch = int(pre_train_model.split("_")[-1][:-4]) + 1
        os.makedirs("exp/{}/{}".format(dataset, save_folder),exist_ok=True)
    else:
        print('Create new model')
        with open("log/{}.txt".format(save_folder), "a+") as f:
            f.write("FLOPs of current model is {}\n".format(flops))
            f.write("Parameters of current model is {}\n".format(params))
        if not os.path.exists("exp/{}/{}".format(dataset, save_folder)):
            try:
                os.mkdir("exp/{}/{}".format(dataset, save_folder))
            except FileNotFoundError:
                os.mkdir("exp/{}".format(dataset))
                os.mkdir("exp/{}/{}".format(dataset, save_folder))

    if optimize == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(),
                                        lr=config.lr,
                                        momentum=config.momentum,
                                        weight_decay=config.weightDecay)
    elif optimize == 'adam':
        optimizer = torch.optim.Adam(
            m.parameters(),
            lr=config.lr,
            weight_decay=config.weightDecay
        )
    else:
        raise Exception

    if mix_precision:
        m, optimizer = amp.initialize(m, optimizer, opt_level="O1")

    writer = SummaryWriter(
        'tensorboard/{}/{}'.format(dataset, save_folder))


    # Model Transfer
    if device != "cpu":
        m = torch.nn.DataParallel(m).cuda()
        criterion = torch.nn.MSELoss().cuda()
    else:
        m = torch.nn.DataParallel(m)
        criterion = torch.nn.MSELoss()

    # Start Training
    for i in range(config.epochs)[begin_epoch:]:
        os.makedirs("log/{}".format(dataset), exist_ok=True)
        log = open("log/{}/{}.txt".format(dataset, save_folder), "a+")
        print('############# Starting Epoch {} #############'.format(i))
        log.write('############# Starting Epoch {} #############\n'.format(i))

        for name, param in m.named_parameters():
            writer.add_histogram(
                name, param.clone().data.to("cpu").numpy(), i)

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

        opt.acc = acc
        opt.loss = loss
        m_dev = m.module

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)

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

        if i % config.save_interval == 0:
            torch.save(
                m_dev.state_dict(), 'exp/{}/{}/model_{}.pkl'.format(dataset, save_folder, i))
            torch.save(
                opt, 'exp/{}/{}/option.pkl'.format(dataset, save_folder, i))
            torch.save(
                optimizer, 'exp/{}/{}/optimizer.pkl'.format(dataset, save_folder))

    writer.close()


if __name__ == '__main__':
    main()

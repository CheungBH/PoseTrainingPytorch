# python train_opt.py --backbone mobilenet --struct huge_bigt --expFolder coco_mobile_pruned --expID 13kps_huge_bigt_DUC2_dpg --trainBatch 32 --validBatch 32 --kps 13 --DUC 2 --addDPG --LR 1e-3

import matplotlib.pyplot as plt
import torch
import time
import cv2
import torch.utils.data
import copy
import csv
import sys
import torch.nn as nn
from dataset.coco_dataset import MyDataset
from tqdm import tqdm
from utils.eval import DataLogger, accuracy
from utils.img import flip, shuffleLR
from src.opt import opt
from tensorboardX import SummaryWriter
import os
import config.config as config
from utils.utils import generate_cmd, lr_decay, get_sparse_value, warm_up_lr, write_csv_title, write_decay_title, \
    write_decay_info, draw_graph
from utils.pytorchtools import EarlyStopping
import shutil
from utils.model_info import print_model_param_flops, print_model_param_nums, get_inference_time
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
open_source_dataset = config.open_source_dataset
warm_up_epoch = max(config.warm_up.keys())
loss_params = config.loss_param
patience_decay = config.patience_decay

# os.makedirs("log/{}".format(dataset), exist_ok=True)

torch.backends.cudnn.benchmark = True


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    pts_acc_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_loss_Loggers = {i: DataLogger() for i in range(opt.kps)}
    m.train()

    train_loader_desc = tqdm(train_loader)
    s = get_sparse_value()
    print("sparse value is {} in epoch {}".format(s, opt.epoch))
    # print("Training")

    for i, (inps, labels, setMask, img_info) in enumerate(train_loader_desc):
        # print("{}".format(img_info[-1]))
        if device != "cpu":
            inps = inps.cuda().requires_grad_()
            labels = labels.cuda()
            setMask = setMask.cuda()
        else:
            inps = inps.requires_grad_()
        out = m(inps)

        # loss = criterion(out.mul(setMask), labels)
        loss = torch.zeros(1).cuda()
        for cons, idx_ls in loss_params.items():
            loss += cons * criterion(out[:, idx_ls, :, :], labels[:, idx_ls, :, :])

        # for idx, logger in pts_loss_Loggers.items():
        #     logger.update(criterion(out.mul(setMask)[:, [idx], :, :], labels[:, [idx], :, :]), inps.size(0))

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        optimizer.zero_grad()

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        for k, v in pts_acc_Loggers.items():
            pts_acc_Loggers[k].update(acc[k+1], inps.size(0))

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # for mod in m.modules():
        #     if isinstance(mod, nn.BatchNorm2d):
        #         mod.weight.grad.data.add_(s * torch.sign(mod.weight.data))

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

    body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
    body_part_loss = [Logger.avg for k, Logger in pts_loss_Loggers.items()]
    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg, body_part_acc, body_part_loss


def valid(val_loader, m, criterion, optimizer, writer):
    drawn_kp, drawn_hm = False, False
    lossLogger = DataLogger()
    accLogger = DataLogger()
    pts_acc_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_loss_Loggers = {i: DataLogger() for i in range(opt.kps)}
    m.eval()

    # print("Validating")

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, img_info) in enumerate(val_loader_desc):
        # print("{}".format(img_info[-1]))
        if device != "cpu":
            inps = inps.cuda()
            labels = labels.cuda()
            setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            if not drawn_kp:
                try:
                    kps_img, have_kp = draw_kps(out, img_info)

                # if have_kp:
                    drawn_kp = True
                except:
                    pass
                writer.add_image("result of epoch {}".format(opt.epoch),
                                 cv2.imread(os.path.join("exp", opt.expFolder, opt.expID, opt.expID, "img.jpg"))[:, :, ::-1],
                                 dataformats='HWC')
                # else:
                #     pass
                drawn_hm = True
                hm = draw_hms(out[0])
                writer.add_image("result of epoch {} --> heatmap".format(opt.epoch), hm)

            loss = criterion(out.mul(setMask), labels)

            # for idx, logger in pts_loss_Loggers.items():
            #     logger.update(criterion(out.mul(setMask)[:,[idx],:,:], labels[:,[idx],:,:]), inps.size(0))

            flip_out = m(flip(inps))
            flip_out = flip(shuffleLR(flip_out, val_loader.dataset))

            out = (flip_out + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        for k, v in pts_acc_Loggers.items():
            pts_acc_Loggers[k].update(acc[k+1], inps.size(0))

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

    body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
    body_part_loss = [Logger.avg for k, Logger in pts_loss_Loggers.items()]
    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg, body_part_acc, body_part_loss


def main():
    cmd_ls = sys.argv[1:]
    cmd = generate_cmd(cmd_ls)
    if "--freeze_bn False" in cmd:
        opt.freeze_bn = False
    if "--addDPG False" in cmd:
        opt.addDPG = False
    print(opt)

    exp_dir = os.path.join("exp/{}/{}".format(dataset, save_folder))
    log_dir = os.path.join(exp_dir, "{}".format(save_folder))
    os.makedirs(log_dir, exist_ok=True)
    log_name = os.path.join(log_dir, "{}.txt".format(save_folder))
    train_log_name = os.path.join(log_dir, "{}_train.xlsx".format(save_folder))
    # Prepare Dataset

    # Model Initialize
    if device != "cpu":
        m = createModel(cfg=model_cfg).cuda()
    else:
        m = createModel(cfg=model_cfg).cpu()
    print(m, file=open("model.txt", "w"))

    begin_epoch = 0
    pre_train_model = opt.loadModel
    flops = print_model_param_flops(m)
    print("FLOPs of current model is {}".format(flops))
    params = print_model_param_nums(m)
    print("Parameters of current model is {}".format(params))
    inf_time = get_inference_time(m, height=opt.outputResH, width=opt.outputResW)
    print("Inference time is {}".format(inf_time))

    if opt.freeze > 0 or opt.freeze_bn:
        if opt.backbone == "mobilenet":
            feature_layer_num = 155
            feature_layer_name = "features"
        elif opt.backbone == "seresnet101":
            feature_layer_num = 327
            feature_layer_name = "preact"
        elif opt.backbone == "shufflenet":
            feature_layer_num = 167
            feature_layer_name = "shuffle"
        else:
            raise ValueError("Not a correct name")

        feature_num = int(opt.freeze * feature_layer_num)

        for idx, (n, p) in enumerate(m.named_parameters()):
            if len(p.shape) == 1 and opt.freeze_bn:
                p.requires_grad = False
            elif feature_layer_name in n and idx < feature_num:
                p.requires_grad = False
            else:
                p.requires_grad = True

    writer = SummaryWriter(
        'tensorboard/{}/{}'.format(dataset, save_folder), comment=cmd)

    if device != "cpu":
        # rnd_inps = Variable(torch.rand(3, 3, 224, 224), requires_grad=True).cuda()
        rnd_inps = torch.rand(3, 3, 224, 224).cuda()
    else:
        rnd_inps = torch.rand(3, 3, 224, 224)
        # rnd_inps = Variable(torch.rand(3, 3, 224, 224), requires_grad=True)
    try:
        writer.add_graph(m, (rnd_inps,))
    except:
        pass

    shuffle_dataset = False
    for k, v in config.train_info.items():
        if k not in open_source_dataset:
            shuffle_dataset = True

    train_dataset = MyDataset(config.train_info, train=True)
    val_dataset = MyDataset(config.train_info, train=False)
    if shuffle_dataset:
        val_dataset.img_val, val_dataset.bbox_val, val_dataset.part_val = \
            train_dataset.img_val, train_dataset.bbox_val, train_dataset.part_val

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

    os.makedirs("exp/{}/{}".format(dataset, save_folder), exist_ok=True)
    if pre_train_model:
        if "duc_se.pth" not in pre_train_model:
            if "pretrain" not in pre_train_model:
                try:
                    info_path = os.path.join("exp", dataset, save_folder, "option.pkl")
                    info = torch.load(info_path)
                    opt.trainIters = info.trainIters
                    opt.valIters = info.valIters
                    begin_epoch = int(pre_train_model.split("_")[-1][:-4]) + 1
                except:
                    # begin_epoch = int(pre_train_model.split("_")[-1][:-4]) + 1
                    with open(log_name, "a+") as f:
                        f.write(cmd)

            print('Loading Model from {}'.format(pre_train_model))
            m.load_state_dict(torch.load(pre_train_model))
        else:
            print('Loading Model from {}'.format(pre_train_model))
            m.load_state_dict(torch.load(pre_train_model))
            os.makedirs("exp/{}/{}".format(dataset, save_folder), exist_ok=True)
    else:
        print('Create new model')
        with open(log_name, "a+") as f:
            f.write(cmd)
            print(opt, file=f)
            f.write("FLOPs of current model is {}\n".format(flops))
            f.write("Parameters of current model is {}\n".format(params))

    with open(os.path.join(log_dir, "tb.py"), "w") as pyfile:
        pyfile.write("import os\n")
        pyfile.write("os.system('conda init bash')\n")
        pyfile.write("os.system('conda activate py36')\n")
        pyfile.write("os.system('tensorboard --logdir=../../../../tensorboard/{}/{}')".format(dataset, save_folder))

    params_to_update, layers = [], 0
    for name, param in m.named_parameters():
        layers += 1
        if param.requires_grad:
            params_to_update.append(param)
            # print("\t", name)
    print("Training {} layers out of {}".format(len(params_to_update), layers))

    if optimize == 'rmsprop':
        optimizer = torch.optim.RMSprop(params_to_update, lr=opt.LR, momentum=opt.momentum, weight_decay=opt.weightDecay)
    elif optimize == 'adam':
        optimizer = torch.optim.Adam(params_to_update, lr=opt.LR, weight_decay=opt.weightDecay)
    elif optimize == 'sgd':
        optimizer = torch.optim.SGD(params_to_update, lr=opt.LR, momentum=opt.momentum, weight_decay=opt.weightDecay)
    else:
        raise Exception

    if mix_precision:
        m, optimizer = amp.initialize(m, optimizer, opt_level="O1")

    # Model Transfer
    if device != "cpu":
        m = torch.nn.DataParallel(m).cuda()
        criterion = torch.nn.MSELoss().cuda()
    else:
        m = torch.nn.DataParallel(m)
        criterion = torch.nn.MSELoss()

    # loss, acc = valid(val_loader, m, criterion, optimizer, writer)
    # print('Valid:-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
    #     idx=-1,
    #     loss=loss,
    #     acc=acc
    # ))

    early_stopping = EarlyStopping(patience=opt.patience, verbose=True)
    train_acc, val_acc, train_loss, val_loss, best_epoch, = 0, 0, float("inf"), float("inf"), 0,
    train_acc_ls, val_acc_ls, train_loss_ls, val_loss_ls, epoch_ls, lr_ls = [], [], [], [], [], []
    decay, decay_epoch, lr, i = 0, [], opt.LR, begin_epoch
    stop = False
    m_best = m

    train_log = open(train_log_name, "w", newline="")
    csv_writer = csv.writer(train_log)
    csv_writer.writerow(write_csv_title())
    begin_time = time.time()

    # Start Training
    for i in range(opt.nEpochs)[begin_epoch:]:

        opt.epoch = i
        epoch_ls.append(i)
        train_log_tmp = [i, lr]

        log = open(log_name, "a+")
        print('############# Starting Epoch {} #############'.format(i))
        log.write('############# Starting Epoch {} #############\n'.format(i))

        # optimizer, lr = adjust_lr(optimizer, i, config.lr_decay, opt.nEpochs)
        # writer.add_scalar("lr", lr, i)
        # print("epoch {}: lr {}".format(i, lr))

        loss, acc, pt_acc, pt_loss = train(train_loader, m, criterion, optimizer, writer)
        train_log_tmp.append(" ")
        train_log_tmp.append(loss)
        train_log_tmp.append(acc.tolist())
        for item in pt_acc:
            train_log_tmp.append(item.tolist())

        train_acc_ls.append(acc)
        train_loss_ls.append(loss)
        train_acc = acc if acc > train_acc else train_acc
        train_loss = loss if loss < train_loss else train_loss

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

        loss, acc, pt_acc, pt_loss = valid(val_loader, m, criterion, optimizer, writer)
        train_log_tmp.append(" ")
        train_log_tmp.insert(5, loss)
        train_log_tmp.insert(6, acc.tolist())
        train_log_tmp.insert(7, " ")
        for item in pt_acc:
            train_log_tmp.append(item.tolist())

        val_acc_ls.append(acc)
        val_loss_ls.append(loss)
        if acc > val_acc:
            best_epoch = i
            val_acc = acc
            torch.save(m_dev.state_dict(), 'exp/{0}/{1}/{1}_best.pkl'.format(dataset, save_folder))
            m_best = copy.deepcopy(m)
        val_loss = loss if loss < val_loss else val_loss

        bn_num = 0
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                bn_num += 1
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
        csv_writer.writerow(train_log_tmp)

        writer.add_scalar("lr", lr, i)
        print("epoch {}: lr {}".format(i, lr))
        lr_ls.append(lr)

        torch.save(
            opt, 'exp/{}/{}/option.pkl'.format(dataset, save_folder, i))
        if i % opt.save_interval == 0 and i != 0:
            torch.save(
                m_dev.state_dict(), 'exp/{0}/{1}/{1}_{2}.pkl'.format(dataset, save_folder, i))

            # torch.save(
            #     optimizer, 'exp/{}/{}/optimizer.pkl'.format(dataset, save_folder))

        if i < warm_up_epoch:
            optimizer, lr = warm_up_lr(optimizer, i)
        elif i == warm_up_epoch:
            lr = opt.LR
            early_stopping(acc)
        else:
            early_stopping(acc)
            if early_stopping.early_stop:
                optimizer, lr = lr_decay(optimizer, lr)
                decay += 1
                # shutil.copy('exp/{0}/{1}/{1}_best.pkl'.format(dataset, save_folder),
                #             'exp/{0}/{1}/{1}_decay{2}_best.pkl'.format(dataset, save_folder, decay))

                if decay > opt.lr_decay_time:
                    stop = True
                else:
                    decay_epoch.append(i)
                    early_stopping.reset(int(opt.patience * patience_decay[decay]))
                    torch.save(
                        m_dev.state_dict(), 'exp/{0}/{1}/{1}_decay{2}.pkl'.format(dataset, save_folder, decay))
                    m = m_best

        for epo, ac in config.bad_epochs.items():
            if i == epo and val_acc < ac:
                stop = True
        if stop:
            print("Training finished at epoch {}".format(i))
            break

    training_time = time.time() - begin_time
    writer.close()
    train_log.close()

    draw_graph(epoch_ls, train_loss_ls, val_loss_ls, train_acc_ls, val_acc_ls, log_dir)

    os.makedirs("result", exist_ok=True)
    result = os.path.join("result", "{}_result_{}.csv".format(opt.expFolder, config.computer))
    exist = os.path.exists(result)

    with open(result, "a+") as f:
        if not exist:
            title_str = "id,backbone,structure,DUC,params,flops,time,loss_param,addDPG,kps,batch_size,optimizer," \
                        "freeze_bn,freeze,sparse,sparse_decay,epoch_num,LR,Gaussian,thresh,weightDecay,loadModel," \
                        "model_location, ,folder_name,train_acc,train_loss,val_acc,val_loss,training_time, " \
                        "best_epoch,final_epoch"
            title_str = write_decay_title(len(decay_epoch), title_str)
            f.write(title_str)
        info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{},{},{},{},{},{}\n".\
            format(save_folder, opt.backbone, opt.struct, opt.DUC, params, flops, inf_time, opt.loss_allocate,
                   opt.addDPG, opt.kps, opt.trainBatch, opt.optMethod, opt.freeze_bn, opt.freeze, opt.sparse_s,
                   opt.sparse_decay,opt.nEpochs, opt.LR, opt.hmGauss, opt.ratio, opt.weightDecay, opt.loadModel,
                   config.computer, os.path.join(opt.expFolder, save_folder), train_acc, train_loss, val_acc, val_loss,
                   training_time, best_epoch, i)
        info_str = write_decay_info(decay_epoch, info_str)
        f.write(info_str)

    # os.makedirs(os.path.join(exp_dir, "graphs"), exist_ok=True)
    #
    # ln1, = plt.plot(epoch_ls[10:], train_loss_ls[10:], color='red', linewidth=3.0, linestyle='--')
    # ln2, = plt.plot(epoch_ls[10:], val_loss_ls[10:], color='blue', linewidth=3.0, linestyle='-.')
    # ln3, = plt.plot(epoch_ls[10:], lr_ls[10:], color='green', linewidth=3.0, linestyle='-')
    # plt.title("Loss")
    # plt.legend(handles=[ln1, ln2, ln3], labels=['train_loss', 'val_loss', "lr"])
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    # ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    # plt.savefig(os.path.join(log_dir, "loss.jpg"))
    # plt.cla()
    #
    # ln1, = plt.plot(epoch_ls, train_acc_ls, color='red', linewidth=3.0, linestyle='--')
    # ln2, = plt.plot(epoch_ls, val_acc_ls, color='blue', linewidth=3.0, linestyle='-.')
    # ln3, = plt.plot(epoch_ls, lr_ls, color='green', linewidth=3.0, linestyle='-')
    #
    # plt.title("Acc")
    # plt.legend(handles=[ln1, ln2, ln3], labels=['train_acc', 'val_acc', "lr"])
    # ax = plt.gca()
    # ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    # ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    # plt.savefig(os.path.join(log_dir, "acc.jpg"))


if __name__ == '__main__':
    main()

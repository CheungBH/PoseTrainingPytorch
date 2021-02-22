# python train_opt.py --backbone mobilenet --struct huge_bigt --expFolder coco_mobile_pruned --expID 13kps_huge_bigt_DUC2_dpg --trainBatch 32 --validBatch 32 --kps 13 --DUC 2 --addDPG --LR 1e-3
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
from utils.eval import DataLogger, accuracy, cal_accuracy, CurveLogger
from utils.img import flip, shuffleLR
from src.opt import opt
from tensorboardX import SummaryWriter
import os
import config.config as config
from utils.utils import generate_cmd, lr_decay, get_sparse_value, warm_up_lr, write_csv_title, write_decay_title, \
    write_decay_info, draw_graph, check_hm, check_part
from utils.pytorchtools import EarlyStopping
from utils.model_info import print_model_param_flops, print_model_param_nums, get_inference_time
from utils.draw import draw_kps, draw_hms


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


try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False


device = config.device
opt.device = device
save_ID = opt.expID
folder = opt.expFolder
optimize = opt.optMethod
open_source_dataset = config.open_source_dataset
warm_up_epoch = max(config.warm_up.keys())
loss_params = config.loss_weight
patience_decay = config.patience_decay
draw_pred_img = False


torch.backends.cudnn.benchmark = True


def train(train_loader, m, criterion, optimizer, writer):
    accLogger, distLogger, lossLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), CurveLogger()
    pts_acc_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_dist_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_curve_Loggers = {i: CurveLogger() for i in range(opt.kps)}

    m.train()

    train_loader_desc = tqdm(train_loader)
    s = get_sparse_value()
    print("sparse value is {} in epoch {}".format(s, opt.epoch))
    # print("Training")

    for i, (inps, labels, setMask, img_info) in enumerate(train_loader_desc):
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
        acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data, train_loader.dataset.accIdxs)
        # acc, exists = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset, img_info[-1])

        optimizer.zero_grad()

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))
        distLogger.update(dist[0], inps.size(0))
        curveLogger.update(maxval.reshape(1,-1).squeeze(), gt.reshape(1,-1).squeeze())
        ave_auc = curveLogger.cal_AUC()
        pr_area = curveLogger.cal_PR()

        for k, v in pts_acc_Loggers.items():
            pts_curve_Loggers[k].update(maxval[k], gt[k])
            if exists[k] > 0:
                pts_acc_Loggers[k].update(acc[k+1], exists[k])
                pts_dist_Loggers[k].update(dist[k+1], exists[k])

        if mix_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if opt.freeze == 0:
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.weight.grad.data.add_(s * torch.sign(mod.weight.data))

        optimizer.step()
        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar('Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar('Train/Acc', accLogger.avg, opt.trainIters)
        writer.add_scalar('Train/Dist', distLogger.avg, opt.trainIters)
        writer.add_scalar('Train/AUC', ave_auc, opt.trainIters)
        writer.add_scalar('Train/PR', pr_area, opt.trainIters)
        
        # TQDM
        train_loader_desc.set_description(
            'Train: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                epoch=opt.epoch,
                loss=lossLogger.avg,
                acc=accLogger.avg * 100,
                dist=distLogger.avg,
                AUC=ave_auc,
                PR=pr_area
            )
        )

    body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
    body_part_dist = [Logger.avg for k, Logger in pts_dist_Loggers.items()]
    body_part_auc = [Logger.cal_AUC() for k, Logger in pts_curve_Loggers.items()]
    body_part_pr = [Logger.cal_PR() for k, Logger in pts_curve_Loggers.items()]
    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg, distLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR(), \
           body_part_acc, body_part_dist, body_part_auc, body_part_pr


def valid(val_loader, m, criterion, writer):
    drawn_kp, drawn_hm = False, False
    accLogger, distLogger, lossLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), CurveLogger()
    pts_acc_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_dist_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_curve_Loggers = {i: CurveLogger() for i in range(opt.kps)}
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
                try:
                    kps_img, have_kp = draw_kps(out, img_info)
                    drawn_kp = True
                    if draw_pred_img:
                        img = cv2.resize(kps_img, (1080, 720))
                        drawn_kp = False
                        cv2.imshow("val_pred", img)
                        cv2.waitKey(0)
                        # a = 1
                        # draw_kps(out, img_info)
                    else:
                        writer.add_image("result of epoch {}".format(opt.epoch),
                                         cv2.imread(
                                             os.path.join("exp", opt.expFolder, opt.expID, opt.expID, "img.jpg"))[:, :,
                                         ::-1], dataformats='HWC')

                        hm = draw_hms(out[0])
                        writer.add_image("result of epoch {} --> heatmap".format(opt.epoch), hm)
                except:
                    pass

            loss = criterion(out.mul(setMask), labels)

            # flip_out = m(flip(inps))
            # flip_out = flip(shuffleLR(flip_out, val_loader.dataset))
            #
            # out = (flip_out + out) / 2

        acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data, val_loader.dataset.accIdxs)
        # acc, exists = accuracy(out.mul(setMask), labels, val_loader.dataset, img_info[-1])

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))
        distLogger.update(dist[0], inps.size(0))
        curveLogger.update(maxval.reshape(1,-1).squeeze(), gt.reshape(1,-1).squeeze())
        ave_auc = curveLogger.cal_AUC()
        pr_area = curveLogger.cal_PR()

        for k, v in pts_acc_Loggers.items():
            pts_curve_Loggers[k].update(maxval[k], gt[k])
            if exists[k] > 0:
                pts_acc_Loggers[k].update(acc[k+1], exists[k])
                pts_dist_Loggers[k].update(dist[k+1], exists[k])

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar('Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar('Valid/Acc', accLogger.avg, opt.valIters)
        writer.add_scalar('Valid/Dist', distLogger.avg, opt.valIters)
        writer.add_scalar('Valid/AUC', ave_auc, opt.valIters)
        writer.add_scalar('Valid/PR', pr_area, opt.valIters)

        val_loader_desc.set_description(
            'Valid: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                epoch=opt.epoch,
                loss=lossLogger.avg,
                acc=accLogger.avg * 100,
                dist=distLogger.avg,
                AUC=ave_auc,
                PR=pr_area
            )
        )

    body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
    body_part_dist = [Logger.avg for k, Logger in pts_dist_Loggers.items()]
    body_part_auc = [Logger.cal_AUC() for k, Logger in pts_curve_Loggers.items()]
    body_part_pr = [Logger.cal_PR() for k, Logger in pts_curve_Loggers.items()]
    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg, distLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR(), \
           body_part_acc, body_part_dist, body_part_auc, body_part_pr


def main():
    cmd_ls = sys.argv[1:]
    cmd = generate_cmd(cmd_ls)
    if "--freeze_bn False" in cmd:
        opt.freeze_bn = False
    if "--addDPG False" in cmd:
        opt.addDPG = False

    print("----------------------------------------------------------------------------------------------------")
    print("This is the model with id {}".format(save_ID))
    print(opt)
    print("Training backbone is: {}".format(opt.backbone))
    dataset_str = ""
    for k, v in config.train_info.items():
        dataset_str += k
        dataset_str += ","
    print("Training data is: {}".format(dataset_str[:-1]))
    print("Warm up end at {}".format(warm_up_epoch))
    for k, v in config.bad_epochs.items():
        if v > 1:
            raise ValueError("Wrong stopping accuracy!")
    print("----------------------------------------------------------------------------------------------------")

    exp_dir = os.path.join("exp/{}/{}".format(folder, save_ID))
    log_dir = os.path.join(exp_dir, "{}".format(save_ID))
    os.makedirs(log_dir, exist_ok=True)
    log_name = os.path.join(log_dir, "{}.txt".format(save_ID))
    train_log_name = os.path.join(log_dir, "{}_train.xlsx".format(save_ID))
    bn_file = os.path.join(log_dir, "{}_bn.txt".format(save_ID))
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
    print("----------------------------------------------------------------------------------------------------")

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

    writer = SummaryWriter('exp/{}/{}'.format(folder, save_ID), comment=cmd)

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

    os.makedirs("exp/{}/{}".format(folder, save_ID), exist_ok=True)
    if pre_train_model:
        if "duc_se.pth" not in pre_train_model:
            if "pretrain" not in pre_train_model:
                try:
                    info_path = os.path.join("exp", folder, save_ID, "option.pkl")
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
            with open(log_name, "a+") as f:
                f.write(cmd)
            print('Loading Model from {}'.format(pre_train_model))
            m.load_state_dict(torch.load(pre_train_model))
            m.conv_out = nn.Conv2d(m.DIM, opt.kps, kernel_size=3, stride=1, padding=1)
            if device != "cpu":
                m.conv_out.cuda()
            os.makedirs("exp/{}/{}".format(folder, save_ID), exist_ok=True)
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
        pyfile.write("os.system('tensorboard --logdir=../../../../exp/{}/{}')".format(folder, save_ID))

    params_to_update, layers = [], 0
    for name, param in m.named_parameters():
        layers += 1
        if param.requires_grad:
            params_to_update.append(param)
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
    train_acc, val_acc, train_loss, val_loss, best_epoch, train_dist, val_dist, train_auc, val_auc, train_PR, val_PR = \
        0, 0, float("inf"), float("inf"), 0, float("inf"), float("inf"), 0, 0, 0, 0
    train_acc_ls, val_acc_ls, train_loss_ls, val_loss_ls, train_dist_ls, val_dist_ls, train_auc_ls, val_auc_ls, \
        train_pr_ls, val_pr_ls, epoch_ls, lr_ls = [], [], [], [], [], [], [], [], [], [], [], []
    decay, decay_epoch, lr, i = 0, [], opt.LR, begin_epoch
    stop = False
    m_best = m

    train_log = open(train_log_name, "w", newline="")
    bn_log = open(bn_file, "w")
    csv_writer = csv.writer(train_log)
    csv_writer.writerow(write_csv_title())
    begin_time = time.time()

    os.makedirs("result", exist_ok=True)
    result = os.path.join("result", "{}_result_{}.csv".format(opt.expFolder, config.computer))
    exist = os.path.exists(result)

    # Start Training
    try:
        for i in range(opt.nEpochs)[begin_epoch:]:

            opt.epoch = i
            epoch_ls.append(i)
            train_log_tmp = [save_ID, i, lr]

            log = open(log_name, "a+")
            print('############# Starting Epoch {} #############'.format(i))
            log.write('############# Starting Epoch {} #############\n'.format(i))

            # optimizer, lr = adjust_lr(optimizer, i, config.lr_decay, opt.nEpochs)
            # writer.add_scalar("lr", lr, i)
            # print("epoch {}: lr {}".format(i, lr))

            loss, acc, dist, auc, pr, pt_acc, pt_dist, pt_auc, pt_pr = \
                train(train_loader, m, criterion, optimizer, writer)
            train_log_tmp.append(" ")
            train_log_tmp.append(loss)
            train_log_tmp.append(acc.tolist())
            train_log_tmp.append(dist.tolist())
            train_log_tmp.append(auc)
            train_log_tmp.append(pr)
            for a in pt_acc:
                train_log_tmp.append(a.tolist())
            train_log_tmp.append(" ")
            for d in pt_dist:
                train_log_tmp.append(d.tolist())
            train_log_tmp.append(" ")
            for ac in pt_auc:
                train_log_tmp.append(ac)
            train_log_tmp.append(" ")
            for p in pt_pr:
                train_log_tmp.append(p)
            train_log_tmp.append(" ")

            train_acc_ls.append(acc)
            train_loss_ls.append(loss)
            train_dist_ls.append(dist)
            train_auc_ls.append(auc)
            train_pr_ls.append(pr)
            train_acc = acc if acc > train_acc else train_acc
            train_loss = loss if loss < train_loss else train_loss
            train_dist = dist if dist < train_dist else train_dist
            train_auc = auc if auc > train_auc else train_auc
            train_PR = pr if pr > train_PR else train_PR

            log.write('Train:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=i,
                    loss=loss,
                    acc=acc,
                    dist=dist,
                    AUC=auc,
                    PR=pr,
                )
            )

            opt.acc = acc
            opt.loss = loss
            m_dev = m.module

            loss, acc, dist, auc, pr, pt_acc, pt_dist, pt_auc, pt_pr = valid(val_loader, m, criterion, writer)
            train_log_tmp.insert(9, loss)
            train_log_tmp.insert(10, acc.tolist())
            train_log_tmp.insert(11, dist.tolist())
            train_log_tmp.insert(12, auc)
            train_log_tmp.insert(13, pr)
            train_log_tmp.insert(14, " ")
            for a in pt_acc:
                train_log_tmp.append(a.tolist())
            train_log_tmp.append(" ")
            for d in pt_dist:
                train_log_tmp.append(d.tolist())
            train_log_tmp.append(" ")
            for ac in pt_auc:
                train_log_tmp.append(ac)
            train_log_tmp.append(" ")
            for p in pt_pr:
                train_log_tmp.append(p)
            train_log_tmp.append(" ")

            val_acc_ls.append(acc)
            val_loss_ls.append(loss)
            val_dist_ls.append(dist)
            val_auc_ls.append(auc)
            val_pr_ls.append(pr)
            if acc > val_acc:
                best_epoch = i
                val_acc = acc
                torch.save(m_dev.state_dict(), 'exp/{0}/{1}/{1}_best_acc.pkl'.format(folder, save_ID))
                m_best = copy.deepcopy(m)
            val_loss = loss if loss < val_loss else val_loss
            if dist < val_dist:
                val_dist = dist
                torch.save(m_dev.state_dict(), 'exp/{0}/{1}/{1}_best_dist.pkl'.format(folder, save_ID))
            if auc > val_auc:
                val_auc = auc
                torch.save(m_dev.state_dict(), 'exp/{0}/{1}/{1}_best_auc.pkl'.format(folder, save_ID))
            if pr > val_PR:
                val_PR = pr
                torch.save(m_dev.state_dict(), 'exp/{0}/{1}/{1}_best_pr.pkl'.format(folder, save_ID))

            log.write('Valid:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=i,
                    loss=loss,
                    acc=acc,
                    dist=dist,
                    AUC=auc,
                    PR=pr,
                )
            )

            bn_sum, bn_num = 0, 0
            for mod in m.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    bn_num += mod.num_features
                    bn_sum += torch.sum(abs(mod.weight))
                    writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), i)

            bn_ave = bn_sum/bn_num
            bn_log.write("{} --> {}".format(i, bn_ave))
            print("Current bn : {} --> {}".format(i, bn_ave))
            bn_log.write("\n")
            log.close()
            csv_writer.writerow(train_log_tmp)

            writer.add_scalar("lr", lr, i)
            print("epoch {}: lr {}".format(i, lr))
            lr_ls.append(lr)

            torch.save(
                opt, 'exp/{}/{}/option.pkl'.format(folder, save_ID, i))
            if i % opt.save_interval == 0 and i != 0:
                torch.save(
                    m_dev.state_dict(), 'exp/{0}/{1}/{1}_{2}.pkl'.format(folder, save_ID, i))
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
                    # if decay == 2:
                    #     draw_pred_img = False
                    if decay > opt.lr_decay_time:
                        stop = True
                    else:
                        decay_epoch.append(i)
                        early_stopping.reset(int(opt.patience * patience_decay[decay]))
                        # torch.save(m_dev.state_dict(), 'exp/{0}/{1}/{1}_decay{2}.pkl'.format(folder, save_ID, decay))
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

        # draw_graph(epoch_ls, train_loss_ls, val_loss_ls, train_acc_ls, val_acc_ls, train_dist_ls, val_dist_ls, log_dir)
        draw_graph(epoch_ls, train_loss_ls, val_loss_ls, "loss", log_dir)
        draw_graph(epoch_ls, train_acc_ls, val_acc_ls, "acc", log_dir)
        draw_graph(epoch_ls, train_auc_ls, val_auc_ls, "AUC", log_dir)
        draw_graph(epoch_ls, train_dist_ls, val_dist_ls, "dist", log_dir)
        draw_graph(epoch_ls, train_pr_ls, val_pr_ls, "PR", log_dir)

        with open(result, "a+") as f:
            if not exist:
                title_str = "id,backbone,structure,DUC,params,flops,time,loss_param,addDPG,kps,batch_size,optimizer," \
                            "freeze_bn,freeze,sparse,sparse_decay,epoch_num,LR,Gaussian,thresh,weightDecay,loadModel," \
                            "model_location, ,folder_name,training_time,train_acc,train_loss,train_dist,train_AUC," \
                            "train_PR,val_acc,val_loss,val_dist,val_AUC,val_PR,best_epoch,final_epoch"
                title_str = write_decay_title(len(decay_epoch), title_str)
                f.write(title_str)
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".\
                format(save_ID, opt.backbone, opt.struct, opt.DUC, params, flops, inf_time, opt.loss_allocate, opt.addDPG,
                       opt.kps, opt.trainBatch, opt.optMethod, opt.freeze_bn, opt.freeze, opt.sparse_s, opt.sparse_decay,
                       opt.nEpochs, opt.LR, opt.hmGauss, opt.ratio, opt.weightDecay, opt.loadModel, config.computer,
                       os.path.join(folder, save_ID), training_time, train_acc, train_loss, train_dist, train_auc,
                       train_PR, val_acc, val_loss, val_dist, val_auc, val_PR, best_epoch, i)
            info_str = write_decay_info(decay_epoch, info_str)
            f.write(info_str)
    except IOError:
        with open(result, "a+") as f:
            training_time = time.time() - begin_time
            writer.close()
            train_log.close()
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{}\n". \
                format(save_ID, opt.backbone, opt.struct, opt.DUC, params, flops, inf_time, opt.loss_allocate, opt.addDPG,
                       opt.kps, opt.trainBatch, opt.optMethod, opt.freeze_bn, opt.freeze, opt.sparse_s, opt.sparse_decay,
                       opt.nEpochs, opt.LR, opt.hmGauss, opt.ratio, opt.weightDecay, opt.loadModel, config.computer,
                       os.path.join(folder, save_ID), training_time, "Some file is closed")
            f.write(info_str)
    except ZeroDivisionError:
        with open(result, "a+") as f:
            training_time = time.time() - begin_time
            writer.close()
            train_log.close()
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{}\n". \
                format(save_ID, opt.backbone, opt.struct, opt.DUC, params, flops, inf_time, opt.loss_allocate, opt.addDPG,
                       opt.kps, opt.trainBatch, opt.optMethod, opt.freeze_bn, opt.freeze, opt.sparse_s, opt.sparse_decay,
                       opt.nEpochs, opt.LR, opt.hmGauss, opt.ratio, opt.weightDecay, opt.loadModel, config.computer,
                       os.path.join(folder, save_ID), training_time, "Gradient flow")
            f.write(info_str)
    except KeyboardInterrupt:
        with open(result, "a+") as f:
            training_time = time.time() - begin_time
            writer.close()
            train_log.close()
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{}\n". \
                format(save_ID, opt.backbone, opt.struct, opt.DUC, params, flops, inf_time, opt.loss_allocate, opt.addDPG,
                       opt.kps, opt.trainBatch, opt.optMethod, opt.freeze_bn, opt.freeze, opt.sparse_s, opt.sparse_decay,
                       opt.nEpochs, opt.LR, opt.hmGauss, opt.ratio, opt.weightDecay, opt.loadModel, config.computer,
                       os.path.join(folder, save_ID), training_time, "Be killed by someone")
            f.write(info_str)

    print("Model {} training finished".format(save_ID))
    print("----------------------------------------------------------------------------------------------------")


if __name__ == '__main__':
    main()

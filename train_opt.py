# python train_opt.py --backbone mobilenet --struct huge_bigt --expFolder coco_mobile_pruned --expID 13kps_huge_bigt_DUC2_dpg --trainBatch 32 --validBatch 32 --kps 13 --DUC 2 --addDPG --LR 1e-3

import matplotlib.pyplot as plt
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


# os.makedirs("log/{}".format(dataset), exist_ok=True)

torch.backends.cudnn.benchmark = True


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)
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
    exp_dir = os.path.join("exp/{}/{}".format(dataset, save_folder))
    log_dir = os.path.join(exp_dir, "{}".format(save_folder))
    os.makedirs(log_dir, exist_ok=True)
    log_name = os.path.join(log_dir, "{}.txt".format(save_folder))
    # Prepare Dataset

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
    inf_time = get_inference_time(m, height=opt.outputResH, width=opt.outputResW)
    print("Inference time is {}".format(inf_time))

    if opt.freeze:
        for n, p in m.named_parameters():
            if "bn" in n:
                p.requires_grad = False
            elif "preact" in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    os.makedirs("exp/{}/{}".format(dataset, save_folder), exist_ok=True)
    if pre_train_model:
        if "duc" not in pre_train_model:
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
            f.write("FLOPs of current model is {}\n".format(flops))
            f.write("Parameters of current model is {}\n".format(params))

    with open(os.path.join(log_dir, "tb.py"), "w") as pyfile:
        pyfile.write("import os\n")
        pyfile.write("os.system('conda init bash')\n")
        pyfile.write("os.system('conda activate py36')\n")
        pyfile.write("os.system('tensorboard --logdir=../../../../tensorboard/{}/{}')".format(dataset, save_folder))

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

    # loss, acc = valid(val_loader, m, criterion, optimizer, writer)
    # print('Valid:-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
    #     idx=-1,
    #     loss=loss,
    #     acc=acc
    # ))

    train_acc, val_acc, train_loss, val_loss, best_epoch = 0, 0, float("inf"), float("inf"), 0
    train_acc_ls, val_acc_ls, train_loss_ls, val_loss_ls, epoch_ls = [], [], [], [], []

    # Start Training
    for i in range(opt.nEpochs)[begin_epoch:]:

        opt.epoch = i
        epoch_ls.append(i)

        log = open(log_name, "a+")
        print('############# Starting Epoch {} #############'.format(i))
        log.write('############# Starting Epoch {} #############\n'.format(i))

        for name, param in m.named_parameters():
            writer.add_histogram(
                name, param.clone().data.to("cpu").numpy(), i)

        optimizer, lr = adjust_lr(optimizer, i, config.lr_decay, opt.nEpochs)
        writer.add_scalar("lr", lr, i)
        print("epoch {}: lr {}".format(i, lr))

        loss, acc = train(train_loader, m, criterion, optimizer, writer)
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

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)
        val_acc_ls.append(acc)
        val_loss_ls.append(loss)
        if acc > val_acc:
            best_epoch = i
            val_acc = acc
            torch.save(
                m_dev.state_dict(), 'exp/{0}/{1}/{1}_best.pkl'.format(dataset, save_folder))
        val_loss = loss if loss < val_loss else val_loss

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
                m_dev.state_dict(), 'exp/{0}/{1}/{1}_{2}.pkl'.format(dataset, save_folder, i))
            torch.save(
                opt, 'exp/{}/{}/option.pkl'.format(dataset, save_folder, i))
            torch.save(
                optimizer, 'exp/{}/{}/optimizer.pkl'.format(dataset, save_folder))

    os.makedirs("result", exist_ok=True)
    result = os.path.join("result", "{}_result.txt".format(opt.expFolder))
    exist = os.path.exists(result)
    with open(result, "a+") as f:
        if not exist:
            f.write("backbone,structure,DUC,params,flops,time,addDPG,kps,batch_size,optimizer,freeze,sparse,epoch_num,"
                    "LR,Gaussian,thresh,weightDecay, ,model_location, folder_name,train_acc,train_loss,val_acc,"
                    "val_loss,best_epoch\n")
        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}," ",{},{},{},{},{},{},{}\n"
                .format(opt.backbone, opt.struct, opt.DUC, params, flops, inf_time, opt.addDPG, opt.kps, opt.trainBatch,
                        opt.optMethod, opt.freeze, opt.sparse_s, opt.nEpochs, opt.LR, opt.hmGauss, opt.ratio,
                        opt.weightDecay, config.computer, os.path.join(opt.expFolder, save_folder), train_acc,
                        train_loss, val_acc, val_loss, best_epoch))

    # os.makedirs(os.path.join(exp_dir, "graphs"), exist_ok=True)

    ln1, = plt.plot(epoch_ls[10:], train_loss_ls[10:], color='red', linewidth=3.0, linestyle='--')
    ln2, = plt.plot(epoch_ls[10:], val_loss_ls[10:], color='blue', linewidth=3.0, linestyle='-.')
    plt.title("Loss")
    plt.legend(handles=[ln1, ln2], labels=['train_loss', 'val_loss'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig(os.path.join(log_dir, "loss.jpg"))
    plt.cla()

    ln1, = plt.plot(epoch_ls, train_acc_ls, color='red', linewidth=3.0, linestyle='--')
    ln2, = plt.plot(epoch_ls, val_acc_ls, color='blue', linewidth=3.0, linestyle='-.')
    plt.title("Acc")
    plt.legend(handles=[ln1, ln2], labels=['train_acc', 'val_acc'])
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    plt.savefig(os.path.join(log_dir, "acc.jpg"))

    writer.close()


if __name__ == '__main__':
    main()

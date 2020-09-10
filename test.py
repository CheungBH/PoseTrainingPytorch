import torch
import torch.utils.data
from dataset.coco_dataset import MyDataset
from tqdm import tqdm
from utils.eval import DataLogger, cal_accuracy, CurveLogger
from src.opt import opt
import config.config as config
from utils.model_info import print_model_param_flops, print_model_param_nums, get_inference_time
from utils.draw import draw_kps
import os

device = config.device
open_source_dataset = config.open_source_dataset
torch.backends.cudnn.benchmark = True


def test(loader, m, criterion):
    accLogger, distLogger, lossLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), CurveLogger()
    pts_acc_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_dist_Loggers = {i: DataLogger() for i in range(opt.kps)}
    pts_curve_Loggers = {i: CurveLogger() for i in range(opt.kps)}
    m.eval()

    test_loader_desc = tqdm(loader)

    for i, (inps, labels, setMask, img_info) in enumerate(test_loader_desc):
        if device != "cpu":
            inps = inps.cuda()
            labels = labels.cuda()
            setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            try:
                draw_kps(out, img_info)
            except:
                pass

            loss = criterion(out.mul(setMask), labels)

        acc, dist, exists, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data, loader.dataset.accIdxs)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))
        distLogger.update(dist[0], inps.size(0))
        curveLogger.update(maxval.reshape(1, -1).squeeze(), gt.reshape(1, -1).squeeze())
        ave_auc = curveLogger.cal_AUC()
        pr_area = curveLogger.cal_PR()

        for k, v in pts_acc_Loggers.items():
            pts_curve_Loggers[k].update(maxval[k], gt[k])
            if exists[k] > 0:
                pts_acc_Loggers[k].update(acc[k + 1], exists[k])
                pts_dist_Loggers[k].update(dist[k + 1], exists[k])

        test_loader_desc.set_description(
            'Test: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
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
    test_loader_desc.close()

    return lossLogger.avg, accLogger.avg, distLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR(), \
           body_part_acc, body_part_dist, body_part_auc, body_part_pr


def test_model(structure, cfg, data_info, weight, batch=4):

    if structure == "mobilenet":
        from models.mobilenet.MobilePose import createModel
        from config.model_cfg import mobile_opt as model_ls
    elif structure == "seresnet101":
        from models.seresnet.FastPose import createModel
        from config.model_cfg import seresnet_cfg as model_ls
    elif structure == "efficientnet":
        from models.efficientnet.EfficientPose import createModel
        from config.model_cfg import efficientnet_cfg as model_ls
    elif structure == "shufflenet":
        from models.shufflenet.ShufflePose import createModel
        from config.model_cfg import shufflenet_cfg as model_ls
    else:
        raise ValueError("Your model name is wrong")
    model_cfg = model_ls[cfg]
    opt.loadModel = weight

    # Model Initialize
    if device != "cpu":
        m = createModel(cfg=model_cfg).cuda()
    else:
        m = createModel(cfg=model_cfg).cpu()

    m.load_state_dict(torch.load(weight))
    flops = print_model_param_flops(m)
    print("FLOPs of current model is {}".format(flops))
    params = print_model_param_nums(m)
    print("Parameters of current model is {}".format(params))
    inf_time = get_inference_time(m, height=opt.outputResH, width=opt.outputResW)
    print("Inference time is {}".format(inf_time))
    print("----------------------------------------------------------------------------------------------------")

    # Model Transfer
    if device != "cpu":
        criterion = torch.nn.MSELoss().cuda()
    else:
        criterion = torch.nn.MSELoss()

    test_dataset = MyDataset(data_info, train=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch, num_workers=4, pin_memory=True)

    loss, acc, dist, auc, pr, pt_acc, pt_dist, pt_auc, pt_pr = test(test_loader, m, criterion)


if __name__ == '__main__':
    data = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}
    weight = "exp/weights/ceiling_5/ceiling_5_best_duc_se.pkl"
    cfg = ""
    backbone = "seresnet101"
    option_path = os.path.join("/".join(weight.split("/")[:-1]), "option.pkl")
    if os.path.exists(option_path):
        info = torch.load(option_path)
        cfg = info.struct
        backbone = info.backbone
        opt.kps = info.kps
    test_model(backbone, cfg, data, weight)

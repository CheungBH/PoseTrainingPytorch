from tqdm import tqdm
import torch
import os
from config.config import device
from trash.dataset.dataloader import TestDataset
from src.opt import opt
from utils.eval import cal_accuracy
from eval.logger import DataLogger, CurveLogger
from utils.train_utils import Criterion
from models.pose_model import PoseModel
from utils.test_utils import check_option_file, list_to_str

criterion = Criterion()
posenet = PoseModel()


class Tester:
    def __init__(self, test_data, model_path, model_cfg=None, print_info=True, batchsize=8, num_worker=1):
        self.test_data = test_data
        self.model_path = model_path
        self.option_file = check_option_file(model_path)
        self.print = print_info
        self.cfg = model_cfg
        self.batch_size = batchsize
        self.num_worker = num_worker

    def build(self, cfg, crit, model_height=256, model_width=256):
        posenet.build(cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        self.crit = crit
        self.build_criterion(self.crit)
        self.height = model_height
        self.width = model_width
        posenet.load(self.model_path)
        opt.kps = self.kps
        self.test_loader = TestDataset(self.test_data).build_dataloader(self.batch_size, self.num_worker, shuffle=False)

    def build_with_opt(self):
        self.load_from_option()
        posenet.build(self.cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        self.build_criterion(self.crit)
        posenet.load(self.model_path)
        opt.kps = self.kps
        self.loader = TestDataset(self.test_data).build_dataloader(self.batch_size, self.num_worker, shuffle=False)

    def test(self):
        accLogger, distLogger, lossLogger, pckhLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
        pts_pckh_Loggers = {i: DataLogger() for i in range(12)}
        self.model.eval()

        test_loader_desc = tqdm(self.loader)

        for i, (inps, labels, setMask, img_info) in enumerate(test_loader_desc):
            if device != "cpu":
                inps = inps.cuda()
                labels = labels.cuda()
                setMask = setMask.cuda()

            with torch.no_grad():
                out = self.model(inps)

                loss = self.criterion(out.mul(setMask), labels)

            acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data, self.loader.dataset.accIdxs)

            accLogger.update(acc[0].item(), inps.size(0))
            lossLogger.update(loss.item(), inps.size(0))
            distLogger.update(dist[0].item(), inps.size(0))
            pckhLogger.update(pckh[0], inps.size(0))
            curveLogger.update(maxval.reshape(1, -1).squeeze(), gt.reshape(1, -1).squeeze())
            ave_auc = curveLogger.cal_AUC()
            pr_area = curveLogger.cal_PR()

            exists = exists.tolist()
            for k, v in pts_acc_Loggers.items():
                pts_curve_Loggers[k].update(maxval[k], gt[k])
                if exists[k] > 0:
                    pts_acc_Loggers[k].update(acc.tolist()[k + 1], exists[k])
                    pts_dist_Loggers[k].update(dist.tolist()[k + 1], exists[k])
            pckh_exist = exists[-12:]
            for k, v in pts_pckh_Loggers.items():
                if exists[k] > 0:
                    pts_pckh_Loggers[k].update(pckh[k + 1], pckh_exist[k])

            test_loader_desc.set_description(
                'Test: | loss: {loss:.8f} | acc: {acc:.2f} | PCKh: {pckh:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                    loss=lossLogger.avg,
                    acc=accLogger.avg * 100,
                    pckh=pckhLogger.avg * 100,
                    dist=distLogger.avg,
                    AUC=ave_auc,
                    PR=pr_area
                )
            )

        self.body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
        self.body_part_dist = [Logger.avg for k, Logger in pts_dist_Loggers.items()]
        self.body_part_auc = [Logger.cal_AUC() for k, Logger in pts_curve_Loggers.items()]
        self.body_part_pr = [Logger.cal_PR() for k, Logger in pts_curve_Loggers.items()]
        self.body_part_pckh = [Logger.avg for k, Logger in pts_pckh_Loggers.items()]
        self.body_part_thresh = [Logger.get_thresh() for k, Logger in pts_curve_Loggers.items()]
        test_loader_desc.close()
        print("----------------------------------------------------------------------------------------------------")

        self.test_loss, self.test_acc, self.test_pckh, self.test_dist, self.test_auc, self.test_pr \
            = lossLogger.avg, accLogger.avg, pckhLogger.avg, distLogger.avg, curveLogger.cal_AUC(), \
              curveLogger.cal_PR()

    def get_benchmark(self):
        self.flops, self.params, self.infer_time = posenet.benchmark()
        if self.print:
            print("FLOPs of current model is {}".format(self.flops))
            print("Parameters of current model is {}".format(self.params))
            print("Inference time is {}".format(self.infer_time))
            print("-------------------------------------------------------------------------------------------------")

    def build_criterion(self, crit):
        self.criterion = criterion.build(crit)

    def load_from_option(self):
        if os.path.exists(self.option_file):
            self.option = torch.load(self.option_file)
            self.height = self.option.inputResH
            self.width = self.option.inputResW
            self.crit = self.option.crit
        else:
            raise FileNotFoundError("The option.pkl doesn't exist! ")

    def save_thresh_to_option(self):
        thresh_str = list_to_str(self.body_part_thresh)
        self.option.thresh = thresh_str
        torch.save(self.option, self.option_file)

    def summarize(self):
        benchmark = [self.flops, self.params, self.infer_time]
        performance = [self.test_acc, self.test_loss, self.test_pckh, self.test_dist, self.test_auc, self.test_pr]
        parts_performance = [self.body_part_pckh, self.body_part_acc, self.body_part_dist, self.body_part_auc,
                             self.body_part_pr]
        return benchmark, performance, parts_performance, self.body_part_thresh


def test_model(model_path, data_info, batchsize=8, num_worker=1, use_option=True, cfg=None, criteria="MSE", height=256,
               width=256):
    from trash.dataset.dataloader import TestDataset
    test_loader = TestDataset(data_info).build_dataloader(batchsize, num_worker, shuffle=False)
    tester = Tester(test_loader, model_path, model_cfg=cfg)
    if use_option:
        tester.build_with_opt()
    else:
        tester.build(cfg, criteria, height, width)
    tester.test()
    tester.get_benchmark()
    benchmark, performance, parts, thresh = tester.summarize()
    # tester.save_thresh_to_option()


if __name__ == '__main__':
    test_data = {"ceiling": ["data/ceiling/0605_new", "data/ceiling/0605_new.h5", 0]}
    model_path = "exp/kps_test/seresnet18/seresnet18_best_acc.pkl"
    model_cfg = "exp/kps_test/seresnet18/data_default.json"
    use_option = True
    tester = Tester(test_data, model_path, model_cfg)
    if use_option:
        tester.build_with_opt()
    else:
        tester.build(model_cfg, "MSE", 256, 256)
    tester.test()
    tester.get_benchmark()
    benchmark, performance, parts, thresh = tester.summarize()

    #test_model(model_path, test_data, cfg=model_cfg)

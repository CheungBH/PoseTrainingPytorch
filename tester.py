from tqdm import tqdm
import torch
import os
from config.config import device

from utils.eval import cal_accuracy
from utils.logger import DataLogger, CurveLogger
from utils.train_utils import Criterion
from models.pose_model import PoseModel
from utils.test_utils import check_option_file, list_to_str

criterion = Criterion()
posenet = PoseModel()


class Tester:
    def __init__(self, test_loader, model_path, print_info=True):
        self.loader = test_loader
        self.model_path = model_path
        self.option_file = check_option_file(model_path)
        self.print = print_info

    def build(self, backbone, kps, cfg, DUC, crit, model_height=256, model_width=256):
        posenet.build(backbone, cfg)
        self.model = posenet.model
        self.crit = crit
        self.build_criterion(self.crit)
        self.kps = kps
        self.height = model_height
        self.width = model_width

    def build_with_opt(self):
        self.load_from_option()
        posenet.build(self.backbone, self.cfg)
        self.model = posenet.model
        self.build_criterion(self.crit)

    def test(self):
        accLogger, distLogger, lossLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
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

            acc, dist, exists, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data, self.loader.dataset.accIdxs)

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
                'Test: | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                    loss=lossLogger.avg,
                    acc=accLogger.avg * 100,
                    dist=distLogger.avg,
                    AUC=ave_auc,
                    PR=pr_area
                )
            )

        self.body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
        self.body_part_dist = [Logger.avg for k, Logger in pts_dist_Loggers.items()]
        self.body_part_auc = [Logger.cal_AUC() for k, Logger in pts_curve_Loggers.items()]
        self.body_part_pr = [Logger.cal_PR() for k, Logger in pts_curve_Loggers.items()]
        self.body_part_thresh = [Logger.get_thresh() for k, Logger in pts_curve_Loggers.items()]
        test_loader_desc.close()
        print("----------------------------------------------------------------------------------------------------")

        self.test_loss, self.test_acc, self.test_dist, self.test_auc, self.test_pr = lossLogger.avg, accLogger.avg, \
                                                                                     distLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR()

    def get_benchmark(self):
        self.flops, self.params, self.infer_time = self.model.benchmark()
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
            self.backbone = self.option.backbone
            self.cfg = self.option.struct
            self.kps = self.option.kps
            self.DUC = self.option.DUC
            self.crit = self.option.crit
        else:
            raise FileNotFoundError("The option.pkl doesn't exist! ")

    def save_thresh_to_option(self):
        thresh_str = list_to_str(self.body_part_thresh)
        self.option.thresh = thresh_str
        torch.save(self.option, self.option_file)

    def summarize(self):
        benchmark = [self.flops, self.params, self.infer_time]
        performance = [self.test_acc, self.test_loss, self.test_dist, self.test_auc, self.test_pr]
        parts_performance = [self.body_part_acc, self.body_part_dist, self.body_part_auc, self.body_part_pr]
        return benchmark, performance, parts_performance, self.body_part_thresh


def test_model(model_path, data_info, batchsize=8, num_worker=1, use_option=True, DUC=0, kps=17,
               backbone="seresnet101", cfg="0", criteria="MSE", height=256, width=256):
    from dataset.loader import TestDataset
    test_loader = TestDataset(data_info).build_dataloader(batchsize, num_worker)
    tester = Tester(test_loader, model_path)
    if use_option:
        tester.build_with_opt()
    else:
        tester.build(backbone, kps, cfg, DUC, criteria, height, width)
    tester.test()
    benchmark, performance, parts, thresh = tester.summarize()
    tester.save_thresh_to_option()


if __name__ == '__main__':
    test_data = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}
    test_model("", test_data)

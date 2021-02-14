import tqdm
import torch
import os
from config.config import device
from collections import defaultdict

from utils.eval import cal_accuracy
from utils.logger import DataLogger
from utils.train_utils import Criterion
from models.pose_model import PoseModel
from utils.test_utils import check_option_file, parse_thresh

criterion = Criterion()
posenet = PoseModel()


class ErrorAnalyser:
    def __init__(self, test_loader, model_path, print_info=True):
        self.loader = test_loader
        self.model_path = model_path
        self.option_file = check_option_file(model_path)
        self.print = print_info
        self.performance = defaultdict(list)

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

    def add_customized_threshold(self):
        if

    def analyser(self):
        accLogger, distLogger, lossLogger = DataLogger(), DataLogger(), DataLogger()
        # pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        # pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        # pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
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

            acc, dist, exists, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data, loader.dataset.accIdxs)

            accLogger.update(acc[0], inps.size(0))
            lossLogger.update(loss.item(), inps.size(0))
            distLogger.update(dist[0], inps.size(0))

            self.performance[img_info[4]] = [acc[0], dist[0], loss.items()]

            # for k, v in pts_acc_Loggers.items():
            #     pts_curve_Loggers[k].update(maxval[k], gt[k])
            #     if exists[k] > 0:
            #         pts_acc_Loggers[k].update(acc[k + 1], exists[k])
            #         pts_dist_Loggers[k].update(dist[k + 1], exists[k])

            test_loader_desc.set_description(
                'Test: | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f}'.format(
                    loss=lossLogger.avg,
                    acc=accLogger.avg * 100,
                    dist=distLogger.avg,
                )
            )

        test_loader_desc.close()
        print("----------------------------------------------------------------------------------------------------")

        self.test_loss, self.test_acc, self.test_dist, self.test_auc, self.test_pr = lossLogger.avg, accLogger.avg, \
                                                                                     distLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR()

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

    def get_valid_joint(self, values, thresholds):
        valid = 0
        for value, threshold in zip(values, thresholds):
            if value < threshold:
                valid += 1
        return valid

    def summarize(self):


def error_analysis(model_path, data_info, batchsize=8, num_worker=1, use_option=False, DUC=0, kps=17,
               backbone="seresnet101", cfg="0", criteria="MSC", height=256, width=256):
    from dataset.loader import TestDataset
    test_loader = TestDataset(data_info).build_dataloader(batchsize, num_worker)
    analyser = ErrorAnalyser(test_loader, model_path)
    if use_option:
        analyser.build_with_opt()
    else:
        analyser.build(backbone, kps, cfg, DUC, criteria, height, width)
    analyser.analyser()
    benchmark, performance, parts, thresh = tester.summarize()


if __name__ == '__main__':
    pass

from tqdm import tqdm
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
    def __init__(self, test_loader, model_path, default_threshold=0.05):
        self.loader = test_loader
        self.model_path = model_path
        self.option_file = check_option_file(model_path)
        self.thresh = default_threshold
        self.performance = defaultdict(list)
        self.max_val_dict = defaultdict(list)
        # self.add_customized_threshold()

    def build(self, backbone, kps, cfg, DUC, crit, model_height=256, model_width=256):
        posenet.build(backbone, cfg)
        self.model = posenet.model
        self.crit = crit
        self.build_criterion(self.crit)
        self.kps = kps
        self.height = model_height
        self.width = model_width
        self.backbone = backbone
        self.cfg = cfg
        self.default_threshold = [self.thresh] * self.kps

    def build_with_opt(self):
        self.load_from_option()
        posenet.build(self.backbone, self.cfg)
        self.model = posenet.model
        self.build_criterion(self.crit)
        self.default_threshold = [self.thresh] * self.kps
        posenet.load(self.model_path)

    # def add_customized_threshold(self):
    #     try:
    #         option = torch.load(self.option_file)
    #         self.customize_threshold = parse_thresh(option.thresh)
    #     except:
    #         self.customize_threshold = None

    def analyse(self):
        accLogger, distLogger, lossLogger = DataLogger(), DataLogger(), DataLogger()
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

            maxval = maxval.t().squeeze().tolist()
            default_valid = self.get_valid_percent(maxval, self.default_threshold)
            performance = [acc[0].tolist(), dist[0].tolist(), loss.tolist(), default_valid]
            if self.customize_threshold:
                customized_valid = self.get_valid_percent(maxval, self.customize_threshold)
                performance.append(customized_valid)

            self.performance[img_info[3][0]] = performance

            test_loader_desc.set_description(
                'Test | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f}'.format(
                    loss=lossLogger.avg,
                    acc=accLogger.avg * 100,
                    dist=distLogger.avg,
                )
            )

        test_loader_desc.close()
        print("----------------------------------------------------------------------------------------------------")

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

    def get_valid_percent(self, values, thresholds):
        valid = 0
        for value, threshold in zip(values, thresholds):
            if value > threshold:
                valid += 1
        return valid/self.kps

    def summarize(self):
        return self.performance


def error_analysis(model_path, data_info, batchsize=1, num_worker=1, use_option=True, DUC=0, kps=17,
               backbone="seresnet101", cfg="0", criteria="MSE", height=256, width=256):
    from dataset.loader import TestDataset
    test_loader = TestDataset(data_info).build_dataloader(batchsize, num_worker)
    analyser = ErrorAnalyser(test_loader, model_path)
    if use_option:
        analyser.build_with_opt()
    else:
        analyser.build(backbone, kps, cfg, DUC, criteria, height, width)
    analyser.analyse()
    performance = analyser.summarize()
    return performance


# class AutoErrorAnalyser:
#     def __init__(self):

if __name__ == '__main__':
    analyse_data = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}
    error = error_analysis("exp/test/default/default_best_acc.pkl", analyse_data)
    print(error)

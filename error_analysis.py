from tqdm import tqdm
import torch
import os
from config.config import device
from dataset.dataloader import TestLoader
from utils.train_utils import Criterion
from models.pose_model import PoseModel
from utils.utils import get_option_path
from eval.evaluator import BatchEvaluator, EpochEvaluator
from utils.test_utils import list_to_str
from collections import defaultdict

posenet = PoseModel()


class ErrorAnalyser:
    out_h, out_w, in_h, in_w, criterion = 64, 64, 256, 256, "MSE"

    def __init__(self, model_cfg, model_path, data_info, data_cfg, option, default_threshold=0.05, write_threshold=False, batchsize=8, num_worker=1):
        if isinstance(data_info, dict):
            self.test_dataset = TestLoader(data_info, data_cfg)
        else:
            self.test_dataset = data_info
            self.test_loader = self.test_dataset.build_dataloader(batchsize, num_worker)
        self.model_path = model_path

        option_file = get_option_path(model_path)
        if os.path.exists(option_file):
            option = torch.load(option_file)
            self.crit = option.crit
            self.out_h, self.out_w, self.in_h, self.in_w = \
                option.output_height, option.output_width, option.input_height, option.input_width

        else:
            self.out_h, self.out_w, self.in_h, self.in_w = \
                self.test_dataset.dataset.out_h, self.test_dataset.dataset.out_w, self.test_dataset.dataset.in_h, \
                self.test_dataset.dataset.in_w

        posenet.build(model_cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        posenet.load(model_path)
        self.criterion = Criterion().build(self.crit)
        self.option = option

        self.part_test_acc, self.part_test_dist, self.part_test_auc, self.part_test_pr, self.part_test_pckh = \
            [], [], [], [], []
        self.batch_size = batchsize
        self.thresh = default_threshold
        self.performance = defaultdict(list)
        self.max_val_dict = defaultdict(list)
        self.write_threshold = write_threshold
        self.default_threshold = [self.thresh] * self.kps

    def get_valid_percent(self, values, thresholds):
        valid = 0
        for value, threshold in zip(values, thresholds):
            if value > threshold:
                valid += 1
        return valid/self.kps

    def add_customized_thresh(self):
        for img_name, max_val in self.max_val_dict.items():
            customized_valid = self.get_valid_percent(max_val, self.customized_thresholds)
            self.performance[img_name].append(customized_valid)

    def save_thresh_to_option(self):
        thresh_str = list_to_str(self.customized_thresholds)
        opt = torch.load(self.option)
        opt.thresh = thresh_str
        torch.save(opt, self.option)

    def analyse(self):
        BatchEval = BatchEvaluator(self.kps, "Test", self.batch_size)
        EpochEval = EpochEvaluator((self.out_h, self.out_w))
        self.model.eval()
        test_loader_desc = tqdm(self.test_loader)

        for i, (inps, labels, meta) in enumerate(test_loader_desc):
            if device != "cpu":
                inps = inps.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                out = self.model(inps)
                loss = self.criterion(out, labels)

            acc, dist, exists, (maxval, valid), (preds, gts) = \
                BatchEval.eval_per_batch(out.data, labels.data, self.out_h)
            BatchEval.update(acc, dist, exists, maxval, valid, loss)
            EpochEval.update(preds, gts, valid.t())

            loss, acc, dist, auc, pr = BatchEval.get_batch_result()

            maxval = maxval.t().squeeze().tolist()
            default_valid = self.get_valid_percent(maxval, self.default_threshold)
            performance = [acc[0].item(), dist[0].item(), loss.item(), default_valid]

            self.performance[meta["path"]] = performance
            self.max_val_dict[meta["path"]] = maxval

            test_loader_desc.set_description(
                'Analysis: {epoch} | loss: {loss:.4f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                    format(epoch=0, loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
            )

        self.body_part_acc, self.body_part_dist, self.body_part_auc, self.body_part_pr = BatchEval.get_kps_result()
        pckh = EpochEval.eval_per_epoch()
        self.test_pckh = pckh[0]
        self.body_part_pckh = pckh[1:]
        self.test_loss, self.test_acc, self.test_dist, self.test_auc, self.test_pr = BatchEval.get_batch_result()

        self.customized_thresholds = [Logger.get_thresh() for k, Logger in BatchEval.pts_curve_Loggers.items()]
        self.add_customized_thresh()
        if self.write_threshold:
            self.save_thresh_to_option()

    # def get_benchmark(self):
    #     self.flops, self.params, self.infer_time = posenet.benchmark()
    #     if self.print:
    #         print("FLOPs of current model is {}".format(self.flops))
    #         print("Parameters of current model is {}".format(self.params))
    #         print("Inference time is {}".format(self.infer_time))
    #         print("-------------------------------------------------------------------------------------------------")

    def summarize(self):
        return self.performance


if __name__ == '__main__':
    data_info = [{"mpii": {"root": "data/mpii",
                          "train_imgs": "MPIIimages",
                          "valid_imgs": "MPIIimages",
                          "test_imgs": "MPIIimages",
                          "train_annot": "mpiitrain_annotonly_train.json",
                          "valid_annot": "mpiitrain_annotonly_test.json",
                          "test_annot": "mpiitrain_annotonly_test.json",
                         }}]
    model_path = "exp/test_kps/aic_13/latest.pth"
    model_cfg = "exp/test_kps/aic_13/model_cfg.json"
    data_cfg = "exp/test_kps/aic_13/data_cfg.json"
    option_path = ""

    analyser = ErrorAnalyser(model_cfg, model_path, data_info, data_cfg, option_path)
    analyser.analyse()
    performance = analyser.summarize()
    print(performance)


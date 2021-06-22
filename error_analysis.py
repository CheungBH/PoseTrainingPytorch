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
from dataset.draw import PredictionVisualizer, HeatmapVisualizer
import cv2

posenet = PoseModel()


class ErrorAnalyser:
    out_h, out_w, in_h, in_w, crit = 64, 64, 256, 256, "MSE"

    def __init__(self, model_cfg, model_path, data_info, data_cfg, dataset_name="coco", default_threshold=0.05,
                 print_info=True, batchsize=1, num_worker=1, draw_preds_img=False):
        if isinstance(data_info, list):
            self.test_dataset = TestLoader(data_info, data_cfg)
        else:
            self.test_dataset = data_info
        self.test_loader = self.test_dataset.build_dataloader(batchsize, num_worker)
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.draw_img = draw_preds_img

        self.option_file = get_option_path(model_path)
        if os.path.exists(self.option_file):
            self.save_thresh = True
            self.option = torch.load(self.option_file)
            self.crit = self.option.crit
            self.out_h, self.out_w, self.in_h, self.in_w = \
                self.option.output_height, self.option.output_width, self.option.input_height, self.option.input_width

        else:
            self.save_thresh = False
            self.out_h, self.out_w, self.in_h, self.in_w = \
                self.test_dataset.dataset.out_h, self.test_dataset.dataset.out_w, self.test_dataset.dataset.in_h, \
                self.test_dataset.dataset.in_w

        posenet.build(model_cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        posenet.load(model_path)
        self.criterion = Criterion().build(self.crit)

        self.part_acc, self.part_dist, self.part_auc, self.part_pr, self.part_pckh = [], [], [], [], []
        self.print = print_info
        self.batch_size = batchsize
        self.default_threshold = [default_threshold] * self.kps
        self.max_val_ls, self.exist_ls = [], []
        self.imgs, self.ids, self.sample_acc, self.sample_loss, self.sample_dist, self.sample_valid_default,\
            self.sample_valid_customized = [], [], [], [], [], [], []

    def analyse(self):
        BatchEval = BatchEvaluator(self.kps, "Test", self.batch_size)
        EpochEval = EpochEvaluator((self.out_h, self.out_w))
        self.model.eval()
        test_loader_desc = tqdm(self.test_loader)

        PV = PredictionVisualizer(self.kps, self.batch_size, self.out_h, self.out_w, self.in_h,
                                       self.in_w, dataset=self.dataset_name, max_img=1)

        for i, (inps, labels, meta) in enumerate(test_loader_desc):

            if True not in (labels > 0):
                continue

            if device != "cpu":
                inps = inps.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                out = self.model(inps)
                loss = self.criterion(out, labels)

            if self.draw_img:
                preds_img = PV.process(out, meta)
                cv2.imshow("pred", preds_img)
                cv2.waitKey(0)

            acc, dist, exists, (maxval, valid), (preds, gts) = \
                BatchEval.eval_per_batch(out.data, labels.data, self.out_h)
            BatchEval.update(acc, dist, exists, maxval, valid, loss)
            EpochEval.update(preds, gts, valid.t())
            maxval = maxval.t().squeeze().tolist()
            default_valid = self.get_valid_percent(maxval, self.default_threshold, exists)
            self.max_val_ls.append(maxval)

            self.imgs += meta["name"]
            self.ids += meta["id"]
            self.sample_acc.append(acc[0].tolist())
            self.sample_loss.append(loss.tolist())
            self.sample_dist.append(dist[0].tolist())
            self.sample_valid_default.append(default_valid)
            self.exist_ls.append(exists)

            loss, acc, dist, auc, pr = BatchEval.get_batch_result()

            test_loader_desc.set_description(
                'Analyser: {epoch} | loss: {loss:.4f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                         format(epoch=0, loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
            )

        self.body_part_acc, self.body_part_dist, self.body_part_auc, self.body_part_pr = BatchEval.get_kps_result()
        pckh = EpochEval.eval_per_epoch()
        self.pckh = pckh[0]
        print("The pckh value of current model is {}".format(self.pckh))
        self.body_part_pckh = pckh[1:]
        self.customized_thresholds = [Logger.get_thresh() for k, Logger in BatchEval.pts_curve_Loggers.items()]
        self.loss, self.acc, self.dist, self.auc, self.pr = BatchEval.get_batch_result()
        self.add_customized_thresh()
        if self.save_thresh:
            self.save_thresh_to_option()

    def get_benchmark(self):
        self.flops, self.params, self.infer_time = posenet.benchmark()
        if self.print:
            print("FLOPs of current model is {}".format(self.flops))
            print("Parameters of current model is {}".format(self.params))
            print("Inference time is {}".format(self.infer_time))

    def save_thresh_to_option(self):
        thresh_str = list_to_str(self.customized_thresholds)
        self.option.thresh = thresh_str
        torch.save(self.option, self.option_file)

    def get_valid_percent(self, values, thresholds, exists):
        valid = 0
        for value, threshold, exist in zip(values, thresholds, exists):
            if exist > 0:
                if value > threshold:
                    valid += 1
            else:
                if value < threshold:
                    valid += 1
        return valid/self.kps

    def add_customized_thresh(self):
        for max_val, exist in zip(self.max_val_ls, self.exist_ls):
            customized_valid = self.get_valid_percent(max_val, self.customized_thresholds, exist)
            self.sample_valid_customized.append(customized_valid)

    def summarize(self):
        return self.imgs, self.ids, self.sample_acc, self.sample_loss, self.sample_dist, self.sample_valid_default,\
            self.sample_valid_customized

    def summarize_test(self):
        benchmark = [self.flops, self.params, self.infer_time]
        performance = [self.acc, self.loss, self.pckh, self.dist, self.auc, self.pr]
        parts_performance = [self.body_part_pckh, self.body_part_acc, self.body_part_dist, self.body_part_auc,
                             self.body_part_pr]
        return benchmark, performance, parts_performance, self.customized_thresholds


if __name__ == '__main__':
    dataset = "ceiling"
    model_path = "exp/test/default/80.pkl"
    model_cfg = "exp/test/default/model_cfg.json"
    data_cfg = "exp/test/default/data_cfg.json"

    from config.config import datasets_info
    data_info = [{dataset: datasets_info[dataset]}]
    analyser = ErrorAnalyser(model_cfg, model_path, data_info, data_cfg, dataset, draw_preds_img=True)
    analyser.analyse()
    item = analyser.summarize()
    print(item)

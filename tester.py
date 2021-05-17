from tqdm import tqdm
import torch
import os
from config.config import device
from dataset.dataloader import TestLoader
from utils.train_utils import Criterion
from models.pose_model import PoseModel
from utils.utils import get_option_path
from eval.evaluator import BatchEvaluator, EpochEvaluator


posenet = PoseModel()


class Tester:
    out_h, out_w, in_h, in_w, criterion = 64, 64, 256, 256, "MSE"

    def __init__(self, model_cfg, model_path, data_info, data_cfg, print_info=True, batchsize=8, num_worker=1):
        if isinstance(data_info, list):
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

        self.part_test_acc, self.part_test_dist, self.part_test_auc, self.part_test_pr, self.part_test_pckh = \
            [], [], [], [], []
        self.print = print_info
        self.batch_size = batchsize

    def test(self):
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
            test_loader_desc.set_description(
                'Test: {epoch} | loss: {loss:.4f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                    format(epoch=0, loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
            )

        self.body_part_acc, self.body_part_dist, self.body_part_auc, self.body_part_pr = BatchEval.get_kps_result()
        pckh = EpochEval.eval_per_epoch()
        self.test_pckh = pckh[0]
        self.body_part_pckh = pckh[1:]
        self.body_part_thresh = [Logger.get_thresh() for k, Logger in BatchEval.pts_curve_Loggers.items()]
        self.test_loss, self.test_acc, self.test_dist, self.test_auc, self.test_pr = BatchEval.get_batch_result()

    def get_benchmark(self):
        self.flops, self.params, self.infer_time = posenet.benchmark()
        if self.print:
            print("FLOPs of current model is {}".format(self.flops))
            print("Parameters of current model is {}".format(self.params))
            print("Inference time is {}".format(self.infer_time))
            print("-------------------------------------------------------------------------------------------------")

    def summarize(self):
        benchmark = [self.flops, self.params, self.infer_time]
        performance = [self.test_acc, self.test_loss, self.test_pckh, self.test_dist, self.test_auc, self.test_pr]
        parts_performance = [self.body_part_pckh, self.body_part_acc, self.body_part_dist, self.body_part_auc,
                             self.body_part_pr]
        return benchmark, performance, parts_performance, self.body_part_thresh


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

    tester = Tester(model_cfg, model_path, data_info, data_cfg)
    tester.test()
    tester.get_benchmark()
    benchmark, performance, parts, thresh = tester.summarize()
    print(performance)
    print(thresh)
    print(parts)


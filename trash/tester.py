from tqdm import tqdm
import torch
import os
from config.config import device
# from trash.dataset.dataloader import TestDataset
from config.opt import opt
from utils.train_utils import Criterion
from models.pose_model import PoseModel
from utils.test_utils import list_to_str
from utils.utils import get_option_path
from eval.evaluator import BatchEvaluator, EpochEvaluator

criterion = Criterion()
posenet = PoseModel()


class Tester:
    def __init__(self, test_data, model_path, model_cfg=None, print_info=True, batchsize=8, num_worker=1):
        self.test_data = test_data
        self.model_path = model_path
        self.option_file = get_option_path(model_path)
        option = torch.load(self.option_file)
        self.out_height, self.out_width = option.out_height, option.out_width
        self.print = print_info
        self.cfg = model_cfg
        self.batch_size = batchsize
        self.num_worker = num_worker
        self.part_test_acc, self.part_test_dist, self.part_test_auc, self.part_test_pr, self.part_test_pckh = \
            [], [], [], [], []

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
        BatchEval = BatchEvaluator(self.kps, "Test", self.batch_size)
        EpochEval = EpochEvaluator((self.out_height, self.out_width))
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
                BatchEval.eval_per_batch(out.data, labels.data, self.out_height)
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
        self.body_part_thresh = [Logger.get_thresh() for k, Logger in BatchEval.curveLogger.items()]
        self.test_loss, self.test_acc, self.test_dist, self.test_auc, self.test_pr = BatchEval.get_batch_result()

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

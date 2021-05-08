from tqdm import tqdm
from eval.evaluator import BatchEvaluator, EpochEvaluator
from config import config
import torch
from utils.train_utils import Criterion, Optimizer, write_csv_title, summary_title
from models.pose_model import PoseModel
import cv2
import os
from tensorboardX import SummaryWriter
import time
# from utils.draw import draw_kps, draw_hms
from dataset.dataloader import TrainLoader
from utils.utils import draw_graph
import csv
import shutil
from dataset.draw import PredictionVisualizer, HeatmapVisualizer

try:
    from apex import amp
    mix_precision = True
except ImportError:
    mix_precision = False
torch.backends.cudnn.benchmark = True

device = config.device
lr_warm_up_dict = config.warm_up
lr_decay_dict = config.lr_decay_dict
stop_dicts = config.bad_epochs
sparse_decay_dict = config.sparse_decay_dict
dataset_info = config.train_info
computer = config.computer
loss_weight = config.loss_weight

criterion = Criterion()
optimizer = Optimizer()
posenet = PoseModel(device=device)


class Trainer:
    def __init__(self, opt, vis_in_training=False):
        #print(opt)
        self.expFolder = os.path.join("exp", opt.expFolder, opt.expID)
        self.opt_path = os.path.join(self.expFolder, "option.pkl")
        self.vis = vis_in_training
        self.curr_epoch = opt.epoch

        os.makedirs(os.path.join(self.expFolder, "logs/images"), exist_ok=True)
        self.tb_writer = SummaryWriter(self.expFolder)
        self.txt_log = os.path.join(self.expFolder, "logs/log.txt")
        self.bn_log = os.path.join(self.expFolder, "logs/bn.txt")
        self.xlsx_log = os.path.join(self.expFolder, "logs/train_xlsx.csv")
        self.summary_log = os.path.join("exp", opt.expFolder, "train_{}-{}.csv".format(opt.expFolder, computer))

        self.build_with_opt(opt)
        self.freeze = False
        self.stop = False
        self.best_epoch = self.curr_epoch

        self.epoch_ls, self.lr_ls, self.bn_mean_ls = [], [], []
        self.train_pckh, self.val_pckh, self.train_pckh_ls, self.val_pckh_ls, self.part_train_pckh, self.part_val_pckh = 0, 0, [], [], [], []
        self.train_acc, self.val_acc, self.train_acc_ls, self.val_acc_ls, self.part_train_acc, self.part_val_acc = 0, 0, [], [], [], []
        self.train_auc, self.val_auc, self.train_auc_ls, self.val_auc_ls, self.part_train_auc, self.part_val_auc = 0, 0, [], [], [], []
        self.train_pr, self.val_pr, self.train_pr_ls, self.val_pr_ls, self.part_train_pr, self.part_val_pr = 0, 0, [], [], [], []
        self.train_dist, self.val_dist, self.train_dist_ls, self.val_dist_ls, self.part_train_dist, self.part_val_dist = float("inf"), float("inf"), [], [], [], []
        self.train_loss, self.val_loss, self.train_loss_ls, self.val_loss_ls, self.part_train_loss, self.part_val_loss = float("inf"), float("inf"), [], [], [], []
        
    def build_with_opt(self, opt):
        self.opt = opt
        self.total_epochs = opt.nEpochs
        self.lr = opt.LR
        self.trainIter, self.valIter = opt.trainIters, opt.valIters

        posenet.init_with_opt(opt)
        self.params_to_update, _ = posenet.get_updating_param()
        self.freeze = posenet.is_freeze
        self.model = posenet.model
        posenet.write_structure(os.path.join(self.expFolder, "logs/model.txt"))

        self.build_criterion(opt.crit)
        self.build_optimizer(opt.optMethod, opt.LR, opt.momentum, opt.weightDecay)
        posenet.model_transfer(device)
        # self.backbone
        self.model = posenet.model
        self.kps, self.backbone, self.se_ratio = posenet.kps, posenet.backbone, posenet.se_ratio
        opt.kps = posenet.kps

        self.dataset = TrainLoader(dataset_info, opt.data_cfg, loss_weight)
        self.loss_weight = {1: [-item for item in range(self.kps + 1)[1:]]}
        self.train_batch, self.val_batch = opt.trainBatch, opt.validBatch
        self.train_loader, self.val_loader = self.dataset.build_dataloader(opt.trainBatch, opt.validBatch,
                                                                           opt.train_worker, opt.val_worker)
        self.inp_height, self.input_width, self.out_height, self.out_width, self.sigma = \
            self.dataset.train_dataset.transform.input_height, self.dataset.train_dataset.transform.input_width, \
            self.dataset.train_dataset.transform.output_height, self.dataset.train_dataset.transform.output_width, \
            self.dataset.train_dataset.transform.sigma

        if opt.lr_schedule == "step":
            from utils.train_utils import StepLRScheduler as scheduler
        else:
            raise ValueError("Scheduler not supported")
        self.lr_scheduler = scheduler(self.total_epochs, lr_warm_up_dict, lr_decay_dict, self.lr)
        self.save_interval = opt.save_interval
        self.build_sparse_scheduler(opt.sparse_s)
        self.flops, self.params, self.inf_time = posenet.benchmark(height=self.inp_height, width=self.input_width)

    def train(self):
        BatchEval = BatchEvaluator(self.kps, "Train", self.opt.trainBatch)
        EpochEval = EpochEvaluator((self.out_height, self.out_width))
        self.model.train()
        train_loader_desc = tqdm(self.train_loader)
        for i, (inps, labels, meta) in enumerate(train_loader_desc):
            # self.stop = True
            if device != "cpu":
                inps = inps.cuda().requires_grad_()
                labels = labels.cuda()
            else:
                inps = inps.requires_grad_()
            out = self.model(inps)

            loss = torch.zeros(1).cuda()
            for cons, idx_ls in self.loss_weight.items():
                loss += cons * self.criterion(out[:, idx_ls, :, :], labels[:, idx_ls, :, :])

            acc, dist, exists, (maxval, valid), (preds, gts) = \
                BatchEval.eval_per_batch(out.data, labels.data, self.out_height)

            EpochEval.update(preds, gts, valid.t())

            self.optimizer.zero_grad()
            BatchEval.update(acc, dist, exists, maxval, valid, loss)

            if mix_precision:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if not self.freeze:
                for mod in self.model.modules():
                    if isinstance(mod, torch.nn.BatchNorm2d):
                       mod.weight.grad.data.add_(self.sparse_s * torch.sign(mod.weight.data))

            self.optimizer.step()
            self.trainIter += 1

            loss, acc, dist, auc, pr = BatchEval.get_batch_result()
            BatchEval.update_tb(self.tb_writer, self.trainIter)
            train_loader_desc.set_description(
                'Train: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                    format(epoch=self.curr_epoch, loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
            )

        body_part_acc, body_part_dist, body_part_auc, body_part_pr = BatchEval.get_kps_result()
        pckh = EpochEval.eval_per_epoch()
        print(pckh)
        self.tb_writer.add_scalar('Train/pckh', pckh[0], self.curr_epoch)

        train_loader_desc.set_description(
            'Train: {epoch} | pckh: {pckh:.4f} | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                format(epoch=self.curr_epoch, pckh=pckh[0], loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
        )
        train_loader_desc.close()

        self.part_train_acc.append(body_part_acc)
        self.part_train_dist.append(body_part_dist)
        self.part_train_auc.append(body_part_auc)
        self.part_train_pr.append(body_part_pr)
        self.part_train_pckh.append(pckh[1:])

        loss, acc, dist, auc, pr = BatchEval.get_batch_result()
        self.update_indicators(acc, loss, dist, pckh[0], auc, pr, self.trainIter, "train")

    def valid(self):
        drawn_kp = False
        PV, HMV = PredictionVisualizer(self.kps, self.val_batch, self.out_height, self.out_width, self.inp_height,
                                       self.input_width), \
                  HeatmapVisualizer(self.out_height, self.out_width)
        BatchEval = BatchEvaluator(self.kps, "Valid", self.opt.validBatch)
        EpochEval = EpochEvaluator((self.out_height, self.out_width))
        self.model.eval()
        val_loader_desc = tqdm(self.val_loader)

        for i, (inps, labels, meta) in enumerate(val_loader_desc):
            if device != "cpu":
                inps = inps.cuda()
                labels = labels.cuda()

            with torch.no_grad():
                out = self.model(inps)

                if not drawn_kp:
                    preds_img = PV.process(out, meta)
                    # hm_img = HMV.draw_hms(out)
                    # self.tb_writer.add_image("result of epoch {} --> heatmap".format(self.curr_epoch), hm_img)

                    cv2.imwrite(os.path.join(self.expFolder, "logs/images/img_{}.jpg".format(self.curr_epoch)), preds_img)
                    self.tb_writer.add_image("result of epoch {}".format(self.curr_epoch), preds_img[:,:,::-1], dataformats='HWC')
                    drawn_kp = True

                loss = torch.zeros(1).cuda()
                for cons, idx_ls in self.loss_weight.items():
                    loss += cons * self.criterion(out[:, idx_ls, :, :], labels[:, idx_ls, :, :])

            acc, dist, exists, (maxval, valid), (preds, gts) = \
                BatchEval.eval_per_batch(out.data, labels.data, self.out_height)
            BatchEval.update(acc, dist, exists, maxval, valid, loss)
            EpochEval.update(preds, gts, valid.t())
            self.valIter += 1

            loss, acc, dist, auc, pr = BatchEval.get_batch_result()
            BatchEval.update_tb(self.tb_writer, self.valIter)
            val_loader_desc.set_description(
                'Valid: {epoch} | loss: {loss:.4f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                    format(epoch=self.curr_epoch, loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
            )

        body_part_acc, body_part_dist, body_part_auc, body_part_pr = BatchEval.get_kps_result()
        pckh = EpochEval.eval_per_epoch()
        print(pckh)
        self.tb_writer.add_scalar('Valid/pckh', pckh[0], self.curr_epoch)

        val_loader_desc.set_description(
            'Valid: {epoch} | pckh: {pckh:.8f} | loss: {loss:.8f} | acc: {acc:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.
                format(epoch=self.curr_epoch, pckh=pckh[0], loss=loss, acc=acc, dist=dist, AUC=auc, PR=pr)
        )

        val_loader_desc.close()
        self.part_val_acc.append(body_part_acc)
        self.part_val_dist.append(body_part_dist)
        self.part_val_auc.append(body_part_auc)
        self.part_val_pr.append(body_part_pr)
        self.part_val_pckh.append(pckh[1:])

        loss, acc, dist, auc, pr = BatchEval.get_batch_result()
        self.update_indicators(acc, loss, dist, pckh[0], auc, pr, self.trainIter, "val")

    def build_criterion(self, crit):
        self.criterion = criterion.build(crit)

    def build_optimizer(self, optim, lr, momen, wd):
        self.optimizer = optimizer.build(optim, self.params_to_update, lr, momen, wd)
        if mix_precision:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def build_sparse_scheduler(self, s):
        self.sparse_s = s
        from utils.train_utils import SparseScheduler
        self.sparse_scheduler = SparseScheduler(self.total_epochs, sparse_decay_dict, self.sparse_s)

    def check_stop(self):
        for stop_epoch, stop_acc in stop_dicts.items():
            if self.curr_epoch == stop_epoch and self.val_acc < stop_acc:
                self.stop = True
                print("The accuracy is too low! Stop.")

    def save(self):
        torch.save(self.opt, self.opt_path)
        torch.save(self.optimizer, '{}/optimizer.pkl'.format(self.expFolder))

        if self.curr_epoch % self.save_interval == 0 and self.curr_epoch != 0:
            torch.save(self.model.module.state_dict(), os.path.join(self.expFolder, "{}.pkl".format(self.curr_epoch)))

    def record_bn(self):
        bn_sum, bn_num = 0, 0
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                bn_num += mod.num_features
                bn_sum += torch.sum(abs(mod.weight))
                self.tb_writer.add_histogram("bn_weight", mod.weight.data.cpu().numpy(), self.curr_epoch)
        bn_ave = bn_sum / bn_num
        self.bn_mean_ls.append(bn_ave)

    def update_indicators(self, acc, loss, dist, pckh, auc, pr, iter, phase):
        if phase == "train":
            self.train_acc_ls.append(acc)
            self.train_loss_ls.append(loss)
            self.train_dist_ls.append(dist)
            self.train_auc_ls.append(auc)
            self.train_pr_ls.append(pr)
            self.train_pckh_ls.append(pckh)
            if acc > self.train_acc:
                self.train_acc = acc
            if pckh > self.train_pckh:
                self.train_pckh = pckh
            if auc > self.train_auc:
                self.train_auc = auc
            if pr > self.train_pr:
                self.train_pr = pr
            if loss < self.train_loss:
                self.train_loss = loss
            if dist < self.train_dist:
                self.train_dist = dist
            self.opt.trainAcc, self.opt.trainLoss, self.opt.trainPCKh, self.opt.trainDist, self.opt.trainAuc, \
                self.opt.trainPR, self.opt.trainIters = acc, loss, pckh, dist, auc, pr, iter
        elif phase == "val":
            self.val_acc_ls.append(acc)
            self.val_loss_ls.append(loss)
            self.val_dist_ls.append(dist)
            self.val_auc_ls.append(auc)
            self.val_pr_ls.append(pr)
            self.val_pckh_ls.append(pckh)
            if acc > self.val_acc:
                self.val_acc = acc
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.expFolder, "{}_best_acc.pkl".format(self.opt.expID)))
                self.best_epoch = self.curr_epoch
            if pckh > self.val_pckh:
                self.val_pckh = pckh
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.expFolder, "{}_best_pckh.pkl".format(self.opt.expID)))
            if auc > self.val_auc:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.expFolder, "{}_best_auc.pkl".format(self.opt.expID)))
                self.val_auc = auc
            if pr > self.val_pr:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.expFolder, "{}_best_pr.pkl".format(self.opt.expID)))
                self.val_pr = pr
            if loss < self.val_loss:
                self.val_loss = loss
            if dist < self.val_dist:
                torch.save(self.model.module.state_dict(),
                           os.path.join(self.expFolder, "{}_best_dist.pkl".format(self.opt.expID)))
                self.val_dist = dist
            self.opt.valAcc, self.opt.valLoss, self.opt.valPCKh, self.opt.valDist, self.opt.valAuc, self.opt.valPR, \
                self.opt.valIters = acc, loss, pckh, dist, auc, pr, iter
        else:
            raise ValueError("The code is wrong!")

    def epoch_result(self, epoch):
        ep_line = [self.opt.expID, self.epoch_ls[epoch], self.lr_ls[epoch], ""]
        ep_performance = [self.train_loss_ls[epoch], self.train_acc_ls[epoch], self.train_pckh_ls[epoch],
                          self.train_dist_ls[epoch], self.train_auc_ls[epoch], self.train_pr_ls[epoch],
                          self.val_loss_ls[epoch], self.val_acc_ls[epoch], self.val_pckh_ls[epoch],
                          self.val_dist_ls[epoch], self.val_auc_ls[epoch], self.val_pr_ls[epoch], ""]
        ep_line += ep_performance
        ep_line += self.part_train_acc[epoch]
        ep_line.append("")
        ep_line += self.part_train_pckh[epoch]
        ep_line.append("")
        ep_line += self.part_train_dist[epoch]
        ep_line.append("")
        ep_line += self.part_train_auc[epoch]
        ep_line.append("")
        ep_line += self.part_train_pr[epoch]
        ep_line.append("")
        ep_line += self.part_val_acc[epoch]
        ep_line.append("")
        ep_line += self.part_val_pckh[epoch]
        ep_line.append("")
        ep_line += self.part_val_dist[epoch]
        ep_line.append("")
        ep_line += self.part_val_auc[epoch]
        ep_line.append("")
        ep_line += self.part_val_pr[epoch]
        ep_line.append("")
        return ep_line

    def draw_graph(self):
        log_dir = os.path.join(self.expFolder, "logs")
        draw_graph(self.epoch_ls, self.train_loss_ls, self.val_loss_ls, "loss", log_dir)
        draw_graph(self.epoch_ls, self.train_acc_ls, self.val_acc_ls, "acc", log_dir)
        draw_graph(self.epoch_ls, self.train_auc_ls, self.val_auc_ls, "AUC", log_dir)
        draw_graph(self.epoch_ls, self.train_dist_ls, self.val_dist_ls, "dist", log_dir)
        draw_graph(self.epoch_ls, self.train_pr_ls, self.val_pr_ls, "PR", log_dir)
        draw_graph(self.epoch_ls, self.train_pckh_ls, self.val_pckh_ls, "PCKh", log_dir)

    def write_xlsx(self):
        with open(self.xlsx_log, "w", newline="") as excel_log:
            csv_writer = csv.writer(excel_log)
            csv_writer.writerow(write_csv_title(self.opt.kps))
            for idx in range(len(self.epoch_ls)):
                csv_writer.writerow(self.epoch_result(idx))

    def write_summary(self, error_str=""):
        exist = False
        if os.path.exists(self.summary_log):
            exist = True
        with open(self.summary_log, "a+") as summary:
            if not exist:
                summary.write(summary_title())
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}, ,{},{},{},{},{}," \
                       "{},{},{},{},{},{},{},{},{},{},{}\n". \
                format(self.opt.expID, self.kps, self.backbone, self.se_ratio, self.params, self.flops, self.inf_time,
                       self.input_width, self.inp_height, self.out_width, self.out_height, self.train_batch,
                       self.opt.optMethod, self.opt.freeze_bn, self.opt.freeze, self.opt.sparse_s, self.total_epochs,
                       self.opt.LR, self.sigma, self.opt.weightDecay, self.opt.loadModel, config.computer,
                       self.expFolder, self.time_elapse, self.train_acc, self.train_loss, self.train_pckh,
                       self.train_dist, self.train_auc, self.train_pr, self.val_acc, self.val_loss, self.val_pckh,
                       self.val_dist, self.val_auc, self.val_pr, self.best_epoch, self.curr_epoch)
            summary.write(info_str + error_str)

    def write_log(self):
        with open(self.bn_log, "a+") as bn_file:
            bn_file.write("Current bn: {} --> {}".format(self.curr_epoch, self.bn_mean_ls[-1]))
            bn_file.write("\n")

        with open(self.txt_log, "a+") as result_file:
            result_file.write('############# Starting Epoch {} #############\n'.format(self.curr_epoch))
            result_file.write('Train:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | PCKh:{pckh: .4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=self.curr_epoch, loss=self.train_loss_ls[-1], acc=self.train_acc_ls[-1], pckh=self.train_pckh_ls[-1],
                    dist=self.train_dist_ls[-1], AUC=self.train_auc_ls[-1], PR=self.train_pr_ls[-1],
                ))
            result_file.write('Valid:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | PCKh:{pckh: .4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=self.curr_epoch, loss=self.val_loss_ls[-1], acc=self.val_acc_ls[-1], pckh=self.val_pckh_ls[-1],
                    dist=self.val_dist_ls[-1], AUC=self.val_auc_ls[-1], PR=self.val_pr_ls[-1],
                ))

    def process(self):
        shutil.copy(self.opt.model_cfg, os.path.join(self.expFolder, "model_cfg.json"))
        shutil.copy(self.opt.data_cfg, os.path.join(self.expFolder, "data_cfg.json"))

        begin_time = time.time()
        error_string = ""
        try:
            for epoch in range(self.total_epochs)[self.curr_epoch:]:
                self.epoch_ls.append(epoch)
                print('############# Starting Epoch {} #############'.format(epoch))

                curr_lr = self.lr_scheduler.update(self.optimizer, epoch)
                self.lr_ls.append(curr_lr)
                self.sparse_s = self.sparse_scheduler.update(epoch)

                self.train()
                self.valid()
                self.record_bn()
                self.write_log()
                self.save()

                self.check_stop()
                if self.stop:
                    error_string = ", The accuracy is too low"
                    break
                self.curr_epoch += 1
        # except IOError:
        #     error_string = ",Some file is closed"
        # except ZeroDivisionError:
        #     error_string = ",Gradient flow"
        except KeyboardInterrupt:
            error_string = ",Process was killed"

        self.time_elapse = time.time() - begin_time
        self.draw_graph()
        self.write_xlsx()
        self.write_summary(error_string)


if __name__ == '__main__':
    from src.opt import opt
    trainer = Trainer(opt)
    trainer.process()

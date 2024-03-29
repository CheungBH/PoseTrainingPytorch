from config import config
import torch
from utils.train_utils import Criterion, Optimizer
from models.pose_model import PoseModel
from utils.pytorchtools import EarlyStopping


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
loss_weight = config.loss_weight
sparse_decay_dict = config.sparse_decay_dict
dataset_info = config.train_info
computer = config.computer

criterion = Criterion()
optimizer = Optimizer()
posenet = PoseModel(device=device)


class Trainer:
    def __init__(self, opt, vis_in_training=False):
        print(opt)
        self.expFolder = os.path.join("exp", opt.expFolder, opt.expID)
        self.opt_path = os.path.join(self.expFolder, "option.pkl")
        self.vis = vis_in_training
        self.curr_epoch = opt.epoch
        self.build_with_opt(opt)

        os.makedirs(os.path.join(self.expFolder, opt.expID), exist_ok=True)
        self.tb_writer = SummaryWriter(self.expFolder)
        self.txt_log = os.path.join(self.expFolder, "{}/log.txt".format(opt.expID))
        self.bn_log = os.path.join(self.expFolder, "{}/bn.txt".format(opt.expID))
        self.xlsx_log = os.path.join(self.expFolder, "{}/train_xlsx.xlsx".format(opt.expID))
        self.summary_log = os.path.join("exp", opt.expFolder, "train_{}.csv".format(opt.expFolder, computer))

        self.freeze = False
        self.stop = False
        self.best_epoch = self.curr_epoch

        self.epoch_ls, self.lr_ls, self.bn_mean_ls = [], [], []
        self.train_pckh, self.val_pckh, self.train_pckh_ls, self.val_pckh_ls, self.part_train_pckh, self.part_val_pckh = 0, 0, [], [], [], []
        self.train_acc, self.val_acc, self.train_acc_ls, self.val_acc_ls, self.part_train_acc, self.part_val_acc = 0, 0, [], [], [], []
        self.train_auc, self.val_auc, self.train_auc_ls, self.val_auc_ls, self.part_train_auc, self.part_val_auc = 0, 0, [], [], [], []
        self.train_pr, self.val_pr, self.train_pr_ls, self.val_pr_ls, self.part_train_pr, self.part_val_pr = 0, 0, [], [], [], []
        self.train_dist, self.val_dist, self.train_dist_ls, self.val_dist_ls, self.part_train_dist, self.part_val_dist = float(
            "inf"), float("inf"), [], [], [], []
        self.train_loss, self.val_loss, self.train_loss_ls, self.val_loss_ls, self.part_train_loss, self.part_val_loss = float(
            "inf"), float("inf"), [], [], [], []

    def build_with_opt(self, opt):
        self.opt = opt
        self.total_epochs = opt.nEpochs
        self.kps = opt.kps
        self.lr = opt.LR
        self.trainIter, self.valIter = opt.trainIters, opt.valIters

        posenet.init_with_opt(opt)
        self.params_to_update, _ = posenet.get_updating_param()
        self.freeze = posenet.is_freeze
        self.model = posenet.model
        self.flops, self.params, self.inf_time = posenet.benchmark(height=self.opt.inputResH, width=self.opt.inputResW)

        self.dataset = TrainDataset(dataset_info, hmGauss=opt.hmGauss, rotate=opt.rotate)
        self.train_loader, self.val_loader = self.dataset.build_dataloader(opt.trainBatch, opt.validBatch,
                                                                           opt.train_worker, opt.val_worker)

        self.build_criterion(opt.crit)
        self.build_optimizer(opt.optMethod, opt.LR, opt.momentum, opt.weightDecay)
        posenet.model_transfer(device)
        self.model = posenet.model

        self.lr_scheduler = EarlyStopping(patience=opt.patience, verbose=True)
        self.loss_weight = loss_weight
        self.save_interval = opt.save_interval
        self.build_sparse_scheduler(opt.sparse_s)


    def train(self):
        accLogger, distLogger, lossLogger, pckhLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
        pts_pckh_Loggers = {i: DataLogger() for i in range(12)}

        self.model.train()
        train_loader_desc = tqdm(self.train_loader)

        for i, (inps, labels, setMask, img_info) in enumerate(train_loader_desc):
            if device != "cpu":
                inps = inps.cuda().requires_grad_()
                labels = labels.cuda()
                setMask = setMask.cuda()
            else:
                inps = inps.requires_grad_()
            out = self.model(inps)

            # loss = criterion(out.mul(setMask), labels)
            loss = torch.zeros(1).cuda()
            for cons, idx_ls in self.loss_weight.items():
                loss += cons * self.criterion(out[:, idx_ls, :, :], labels[:, idx_ls, :, :])

            # for idx, logger in pts_loss_Loggers.items():
            #     logger.update(criterion(out.mul(setMask)[:, [idx], :, :], labels[:, [idx], :, :]), inps.size(0))
            acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data,
                                                                 self.train_loader.dataset.accIdxs)
            # acc, exists = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset, img_info[-1])

            self.optimizer.zero_grad()

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
            # Tensorboard
            self.tb_writer.add_scalar('Train/Loss', lossLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/Acc', accLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/Dist', distLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/pckh', distLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/AUC', ave_auc, self.trainIter)
            self.tb_writer.add_scalar('Train/PR', pr_area, self.trainIter)

            # TQDM
            train_loader_desc.set_description(
                'Train: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | PCKh: {pckh:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                    epoch=self.curr_epoch,
                    loss=lossLogger.avg,
                    acc=accLogger.avg * 100,
                    pckh=pckhLogger.avg * 100,
                    dist=distLogger.avg,
                    AUC=ave_auc,
                    PR=pr_area
                )
            )

        body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
        body_part_dist = [Logger.avg for k, Logger in pts_dist_Loggers.items()]
        body_part_auc = [Logger.cal_AUC() for k, Logger in pts_curve_Loggers.items()]
        body_part_pr = [Logger.cal_PR() for k, Logger in pts_curve_Loggers.items()]
        body_part_pckh = [Logger.avg.tolist() for k, Logger in pts_pckh_Loggers.items()]
        train_loader_desc.close()

        self.part_train_acc.append(body_part_acc)
        self.part_train_dist.append(body_part_dist)
        self.part_train_auc.append(body_part_auc)
        self.part_train_pr.append(body_part_pr)
        self.part_train_pckh.append(body_part_pckh)

        curr_acc, curr_loss, curr_dist, curr_pckh, curr_auc, curr_pr = accLogger.avg, lossLogger.avg, distLogger.avg, \
                                                                       pckhLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR()
        self.update_indicators(curr_acc, curr_loss, curr_dist, curr_pckh, curr_auc, curr_pr, self.trainIter, "train")

    def valid(self):
        drawn_kp, drawn_hm = False, False
        accLogger, distLogger, lossLogger, pckhLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_pckh_Loggers = {i: DataLogger() for i in range(12)}
        pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
        self.model.eval()

        val_loader_desc = tqdm(self.val_loader)

        for i, (inps, labels, setMask, img_info) in enumerate(val_loader_desc):
            if device != "cpu":
                inps = inps.cuda()
                labels = labels.cuda()
                setMask = setMask.cuda()

            with torch.no_grad():
                out = self.model(inps)

                if not drawn_kp:
                    try:
                        kps_img, have_kp = draw_kps(out, img_info)
                        drawn_kp = True
                        if self.vis:
                            img = cv2.resize(kps_img, (1080, 720))
                            drawn_kp = False
                            cv2.imshow("val_pred", img)
                            cv2.waitKey(0)
                            # a = 1
                            # draw_kps(out, img_info)
                        else:
                            self.tb_writer.add_image("result of epoch {}".format(self.curr_epoch),
                                                     cv2.imread(
                                                         os.path.join(self.expFolder, "assets/img.jpg"))[:, :, ::-1],
                                                     dataformats='HWC')
                            hm = draw_hms(out[0])
                            self.tb_writer.add_image("result of epoch {} --> heatmap".format(self.curr_epoch), hm)
                    except:
                        pass

                loss = self.criterion(out.mul(setMask), labels)

                # flip_out = m(flip(inps))
                # flip_out = flip(shuffleLR(flip_out, val_loader.dataset))
                #
                # out = (flip_out + out) / 2

            acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data,
                                                                 self.val_loader.dataset.accIdxs)
            # acc, exists = accuracy(out.mul(setMask), labels, val_loader.dataset, img_info[-1])

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

            self.valIter += 1
            # Tensorboard
            self.tb_writer.add_scalar('Valid/Loss', lossLogger.avg, self.valIter)
            self.tb_writer.add_scalar('Valid/Acc', accLogger.avg, self.valIter)
            self.tb_writer.add_scalar('Valid/Dist', distLogger.avg, self.valIter)
            self.tb_writer.add_scalar('Valid/AUC', ave_auc, self.valIter)
            self.tb_writer.add_scalar('Valid/PR', pr_area, self.valIter)
            self.tb_writer.add_scalar('Valid/PCKh', pckhLogger.avg, self.valIter)

            val_loader_desc.set_description(
                'Valid: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | PCKh: {pckh:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                    epoch=self.curr_epoch,
                    loss=lossLogger.avg,
                    pckh=pckhLogger.avg * 100,
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
        body_part_pckh = [Logger.avg for k, Logger in pts_pckh_Loggers.items()]
        val_loader_desc.close()

        self.part_val_acc.append(body_part_acc)
        self.part_val_dist.append(body_part_dist)
        self.part_val_auc.append(body_part_auc)
        self.part_val_pr.append(body_part_pr)
        self.part_val_pckh.append(body_part_pckh)

        curr_acc, curr_loss, curr_pckh, curr_dist, curr_auc, curr_pr = accLogger.avg, lossLogger.avg, distLogger.avg, \
                                                                       pckhLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR()
        self.update_indicators(curr_acc, curr_loss, curr_dist, curr_pckh, curr_auc, curr_pr, self.valIter, "val")

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
            self.train_acc_ls.append(acc.tolist())
            self.train_loss_ls.append(loss)
            self.train_dist_ls.append(dist.tolist())
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
            self.val_acc_ls.append(acc.tolist())
            self.val_loss_ls.append(loss)
            self.val_dist_ls.append(dist.tolist())
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
        log_dir = os.path.join(self.expFolder, self.opt.expID)
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
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}{},{},{},{},{},{},{},{}, ,{},{},{},{},{}," \
                       "{},{},{},{},{},{},{},{},{},{},{}\n". \
                format(self.opt.expID, self.opt.backbone, self.opt.struct, self.opt.se_ratio, self.opt.DUC,
                       self.opt.inputResH, self.opt.inputResW, self.params, self.flops, self.inf_time,
                       self.opt.loss_weight, self.opt.addDPG, self.opt.kps, self.opt.trainBatch, self.opt.optMethod,
                       self.opt.freeze_bn, self.opt.freeze, self.opt.sparse_s, self.opt.nEpochs, self.opt.LR,
                       self.opt.hmGauss, self.opt.ratio, self.opt.weightDecay, self.opt.loadModel, config.computer,
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
            result_file.write(
                'Train:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=self.curr_epoch, loss=self.train_loss_ls[-1], acc=self.train_acc_ls[-1],
                    dist=self.train_dist_ls[-1], AUC=self.train_auc_ls[-1], PR=self.train_pr_ls[-1],
                ))
            result_file.write(
                'Valid:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=self.curr_epoch, loss=self.val_loss_ls[-1], acc=self.val_acc_ls[-1],
                    dist=self.val_dist_ls[-1], AUC=self.val_auc_ls[-1], PR=self.val_pr_ls[-1],
                ))

    def process(self):
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
        except IOError:
            error_string = ",Some file is closed"
        except ZeroDivisionError:
            error_string = ",Gradient flow"
        except KeyboardInterrupt:
            error_string = ",Process was killed"

        self.time_elapse = time.time() - begin_time
        self.draw_graph()
        self.write_xlsx()
        self.write_summary(error_string)


if __name__ == '__main__':
    from config.opt import opt

    trainer = Trainer(opt)
    trainer.process()
from tqdm import tqdm
from eval.logger import DataLogger, CurveLogger
from utils.eval import cal_accuracy
from config import config
import torch
from utils.train_utils import Criterion, Optimizer, write_csv_title, summary_title
from models.pose_model import PoseModel
import cv2
import os
from tensorboardX import SummaryWriter
import time
from utils.draw import draw_kps, draw_hms
from trash.dataset.dataloader import TrainDataset
from utils.utils import draw_graph
import csv

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
loss_weight = config.loss_weight
sparse_decay_dict = config.sparse_decay_dict
dataset_info = config.train_info
computer = config.computer

criterion = Criterion()
optimizer = Optimizer()
posenet = PoseModel(device=device)


class Trainer:
    def __init__(self, opt, vis_in_training=False):
        print(opt)
        self.expFolder = os.path.join("exp", opt.expFolder, opt.expID)
        self.opt_path = os.path.join(self.expFolder, "option.pkl")
        self.vis = vis_in_training
        self.curr_epoch = opt.epoch
        self.build_with_opt(opt)

        os.makedirs(os.path.join(self.expFolder, opt.expID), exist_ok=True)
        self.tb_writer = SummaryWriter(self.expFolder)
        self.txt_log = os.path.join(self.expFolder, "{}/log.txt".format(opt.expID))
        self.bn_log = os.path.join(self.expFolder, "{}/bn.txt".format(opt.expID))
        self.xlsx_log = os.path.join(self.expFolder, "{}/train_xlsx.xlsx".format(opt.expID))
        self.summary_log = os.path.join("exp", opt.expFolder, "train_{}.csv".format(opt.expFolder, computer))

        self.freeze = False
        self.stop = False
        self.best_epoch = self.curr_epoch

        self.epoch_ls, self.lr_ls, self.bn_mean_ls = [], [], []
        self.train_pckh, self.val_pckh, self.train_pckh_ls, self.val_pckh_ls, self.part_train_pckh, self.part_val_pckh = 0, 0, [], [], [], []
        self.train_acc, self.val_acc, self.train_acc_ls, self.val_acc_ls, self.part_train_acc, self.part_val_acc = 0, 0, [], [], [], []
        self.train_auc, self.val_auc, self.train_auc_ls, self.val_auc_ls, self.part_train_auc, self.part_val_auc = 0, 0, [], [], [], []
        self.train_pr, self.val_pr, self.train_pr_ls, self.val_pr_ls, self.part_train_pr, self.part_val_pr = 0, 0, [], [], [], []
        self.train_dist, self.val_dist, self.train_dist_ls, self.val_dist_ls, self.part_train_dist, self.part_val_dist = float(
            "inf"), float("inf"), [], [], [], []
        self.train_loss, self.val_loss, self.train_loss_ls, self.val_loss_ls, self.part_train_loss, self.part_val_loss = float(
            "inf"), float("inf"), [], [], [], []

    def build_with_opt(self, opt):
        self.opt = opt
        self.total_epochs = opt.nEpochs
        self.kps = opt.kps
        self.lr = opt.LR
        self.trainIter, self.valIter = opt.trainIters, opt.valIters

        posenet.init_with_opt(opt)
        self.params_to_update, _ = posenet.get_updating_param()
        self.freeze = posenet.is_freeze
        self.model = posenet.model
        self.flops, self.params, self.inf_time = posenet.benchmark(height=self.opt.inputResH, width=self.opt.inputResW)

        self.dataset = TrainDataset(dataset_info, hmGauss=opt.hmGauss, rotate=opt.rotate)
        self.train_loader, self.val_loader = self.dataset.build_dataloader(opt.trainBatch, opt.validBatch,
                                                                           opt.train_worker, opt.val_worker)

        self.build_criterion(opt.crit)
        self.build_optimizer(opt.optMethod, opt.LR, opt.momentum, opt.weightDecay)
        posenet.model_transfer(device)
        self.model = posenet.model

        if opt.lr_schedule == "step":
            from utils.train_utils import StepLRScheduler as scheduler
        else:
            raise ValueError("Scheduler not supported")
        self.lr_scheduler = scheduler(self.total_epochs, lr_warm_up_dict, lr_decay_dict, self.lr)
        self.loss_weight = loss_weight
        self.save_interval = opt.save_interval
        self.build_sparse_scheduler(opt.sparse_s)

    def train(self):
        accLogger, distLogger, lossLogger, pckhLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
        pts_pckh_Loggers = {i: DataLogger() for i in range(12)}

        self.model.train()
        train_loader_desc = tqdm(self.train_loader)

        for i, (inps, labels, setMask, img_info) in enumerate(train_loader_desc):
            if device != "cpu":
                inps = inps.cuda().requires_grad_()
                labels = labels.cuda()
                setMask = setMask.cuda()
            else:
                inps = inps.requires_grad_()
            out = self.model(inps)

            # loss = criterion(out.mul(setMask), labels)
            loss = torch.zeros(1).cuda()
            for cons, idx_ls in self.loss_weight.items():
                loss += cons * self.criterion(out[:, idx_ls, :, :], labels[:, idx_ls, :, :])

            # for idx, logger in pts_loss_Loggers.items():
            #     logger.update(criterion(out.mul(setMask)[:, [idx], :, :], labels[:, [idx], :, :]), inps.size(0))
            acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data,
                                                                 self.train_loader.dataset.accIdxs)
            # acc, exists = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset, img_info[-1])

            self.optimizer.zero_grad()

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
            # Tensorboard
            self.tb_writer.add_scalar('Train/Loss', lossLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/Acc', accLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/Dist', distLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/pckh', distLogger.avg, self.trainIter)
            self.tb_writer.add_scalar('Train/AUC', ave_auc, self.trainIter)
            self.tb_writer.add_scalar('Train/PR', pr_area, self.trainIter)

            # TQDM
            train_loader_desc.set_description(
                'Train: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | PCKh: {pckh:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                    epoch=self.curr_epoch,
                    loss=lossLogger.avg,
                    acc=accLogger.avg * 100,
                    pckh=pckhLogger.avg * 100,
                    dist=distLogger.avg,
                    AUC=ave_auc,
                    PR=pr_area
                )
            )

        body_part_acc = [Logger.avg for k, Logger in pts_acc_Loggers.items()]
        body_part_dist = [Logger.avg for k, Logger in pts_dist_Loggers.items()]
        body_part_auc = [Logger.cal_AUC() for k, Logger in pts_curve_Loggers.items()]
        body_part_pr = [Logger.cal_PR() for k, Logger in pts_curve_Loggers.items()]
        body_part_pckh = [Logger.avg.tolist() for k, Logger in pts_pckh_Loggers.items()]
        train_loader_desc.close()

        self.part_train_acc.append(body_part_acc)
        self.part_train_dist.append(body_part_dist)
        self.part_train_auc.append(body_part_auc)
        self.part_train_pr.append(body_part_pr)
        self.part_train_pckh.append(body_part_pckh)

        curr_acc, curr_loss, curr_dist, curr_pckh, curr_auc, curr_pr = accLogger.avg, lossLogger.avg, distLogger.avg, \
                                                                       pckhLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR()
        self.update_indicators(curr_acc, curr_loss, curr_dist, curr_pckh, curr_auc, curr_pr, self.trainIter, "train")

    def valid(self):
        drawn_kp, drawn_hm = False, False
        accLogger, distLogger, lossLogger, pckhLogger, curveLogger = DataLogger(), DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        pts_acc_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_dist_Loggers = {i: DataLogger() for i in range(self.kps)}
        pts_pckh_Loggers = {i: DataLogger() for i in range(12)}
        pts_curve_Loggers = {i: CurveLogger() for i in range(self.kps)}
        self.model.eval()

        val_loader_desc = tqdm(self.val_loader)

        for i, (inps, labels, setMask, img_info) in enumerate(val_loader_desc):
            if device != "cpu":
                inps = inps.cuda()
                labels = labels.cuda()
                setMask = setMask.cuda()

            with torch.no_grad():
                out = self.model(inps)

                if not drawn_kp:
                    try:
                        kps_img, have_kp = draw_kps(out, img_info)
                        drawn_kp = True
                        if self.vis:
                            img = cv2.resize(kps_img, (1080, 720))
                            drawn_kp = False
                            cv2.imshow("val_pred", img)
                            cv2.waitKey(0)
                            # a = 1
                            # draw_kps(out, img_info)
                        else:
                            self.tb_writer.add_image("result of epoch {}".format(self.curr_epoch),
                                                     cv2.imread(
                                                         os.path.join(self.expFolder, "assets/img.jpg"))[:, :, ::-1],
                                                     dataformats='HWC')
                            hm = draw_hms(out[0])
                            self.tb_writer.add_image("result of epoch {} --> heatmap".format(self.curr_epoch), hm)
                    except:
                        pass

                loss = self.criterion(out.mul(setMask), labels)

                # flip_out = m(flip(inps))
                # flip_out = flip(shuffleLR(flip_out, val_loader.dataset))
                #
                # out = (flip_out + out) / 2

            acc, dist, exists, pckh, (maxval, gt) = cal_accuracy(out.data.mul(setMask), labels.data,
                                                                 self.val_loader.dataset.accIdxs)
            # acc, exists = accuracy(out.mul(setMask), labels, val_loader.dataset, img_info[-1])

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

            self.valIter += 1
            # Tensorboard
            self.tb_writer.add_scalar('Valid/Loss', lossLogger.avg, self.valIter)
            self.tb_writer.add_scalar('Valid/Acc', accLogger.avg, self.valIter)
            self.tb_writer.add_scalar('Valid/Dist', distLogger.avg, self.valIter)
            self.tb_writer.add_scalar('Valid/AUC', ave_auc, self.valIter)
            self.tb_writer.add_scalar('Valid/PR', pr_area, self.valIter)
            self.tb_writer.add_scalar('Valid/PCKh', pckhLogger.avg, self.valIter)

            val_loader_desc.set_description(
                'Valid: {epoch} | loss: {loss:.8f} | acc: {acc:.2f} | PCKh: {pckh:.2f} | dist: {dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}'.format(
                    epoch=self.curr_epoch,
                    loss=lossLogger.avg,
                    pckh=pckhLogger.avg * 100,
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
        body_part_pckh = [Logger.avg for k, Logger in pts_pckh_Loggers.items()]
        val_loader_desc.close()

        self.part_val_acc.append(body_part_acc)
        self.part_val_dist.append(body_part_dist)
        self.part_val_auc.append(body_part_auc)
        self.part_val_pr.append(body_part_pr)
        self.part_val_pckh.append(body_part_pckh)

        curr_acc, curr_loss, curr_pckh, curr_dist, curr_auc, curr_pr = accLogger.avg, lossLogger.avg, distLogger.avg, \
                                                                       pckhLogger.avg, curveLogger.cal_AUC(), curveLogger.cal_PR()
        self.update_indicators(curr_acc, curr_loss, curr_dist, curr_pckh, curr_auc, curr_pr, self.valIter, "val")

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
            self.train_acc_ls.append(acc.tolist())
            self.train_loss_ls.append(loss)
            self.train_dist_ls.append(dist.tolist())
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
            self.val_acc_ls.append(acc.tolist())
            self.val_loss_ls.append(loss)
            self.val_dist_ls.append(dist.tolist())
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
        log_dir = os.path.join(self.expFolder, self.opt.expID)
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
            info_str = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}{},{},{},{},{},{},{},{}, ,{},{},{},{},{}," \
                       "{},{},{},{},{},{},{},{},{},{},{}\n". \
                format(self.opt.expID, self.opt.backbone, self.opt.struct, self.opt.se_ratio, self.opt.DUC,
                       self.opt.inputResH, self.opt.inputResW, self.params, self.flops, self.inf_time,
                       self.opt.loss_weight, self.opt.addDPG, self.opt.kps, self.opt.trainBatch, self.opt.optMethod,
                       self.opt.freeze_bn, self.opt.freeze, self.opt.sparse_s, self.opt.nEpochs, self.opt.LR,
                       self.opt.hmGauss, self.opt.ratio, self.opt.weightDecay, self.opt.loadModel, config.computer,
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
            result_file.write(
                'Train:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=self.curr_epoch, loss=self.train_loss_ls[-1], acc=self.train_acc_ls[-1],
                    dist=self.train_dist_ls[-1], AUC=self.train_auc_ls[-1], PR=self.train_pr_ls[-1],
                ))
            result_file.write(
                'Valid:{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f} | dist:{dist:.4f} | AUC: {AUC:.4f} | PR: {PR:.4f}\n'.format(
                    idx=self.curr_epoch, loss=self.val_loss_ls[-1], acc=self.val_acc_ls[-1],
                    dist=self.val_dist_ls[-1], AUC=self.val_auc_ls[-1], PR=self.val_pr_ls[-1],
                ))

    # def adjust_lr(self):
    #     if self.curr_epoch < max(config.warm_up.keys()):
    #         optimizer, lr =

    def process(self):
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
        except IOError:
            error_string = ",Some file is closed"
        except ZeroDivisionError:
            error_string = ",Gradient flow"
        except KeyboardInterrupt:
            error_string = ",Process was killed"

        self.time_elapse = time.time() - begin_time
        self.draw_graph()
        self.write_xlsx()
        self.write_summary(error_string)


if __name__ == '__main__':
    from config.opt import opt

    trainer = Trainer(opt)
    trainer.process()

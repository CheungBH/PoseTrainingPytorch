from eval.logger import DataLogger, CurveLogger
from eval.utils import *
# from eval.pckh import PCKHCalculator


class BatchEvaluator:
    def __init__(self, kps, phase, bs):
        self.phase = phase
        self.kps = kps
        self.batch_size = bs
        self.accLogger, self.distLogger, self.lossLogger, self.curveLogger = DataLogger(), DataLogger(), DataLogger(), \
                                                                             CurveLogger()
        self.pts_acc_Loggers = {i: DataLogger() for i in range(kps)}
        self.pts_dist_Loggers = {i: DataLogger() for i in range(kps)}
        self.pts_curve_Loggers = {i: CurveLogger() for i in range(kps)}

    def eval_per_batch(self, output, label, out_height):
        label, output = label.cpu().data, output.cpu().data
        preds, preds_maxval = getPreds(output)
        gt, _ = getPreds(label)

        if_exist = torch.Tensor([torch.sum((label[i][j] > 0).float()) > 0 for i in range(len(label))
                                 for j in range(len(label[0]))]).view(len(label), len(label[0])).t()

        norm = torch.ones(preds.size(0)) * out_height / 10
        dists = calc_dists(preds, gt, norm)
        acc, sum_dist, exist = torch.zeros(self.kps + 1), torch.zeros(self.kps + 1), torch.zeros(self.kps)
        # pckh = cal_pckh(gt, preds, if_exist.t(), refp=0.5)

        for i, kps_dist in enumerate(dists):
            nums = exist_id(if_exist[i])
            exist[i] = len(nums)
            if len(nums) > 0:
                dist = kps_dist[nums]
                sum_dist[i + 1] = torch.sum(dist) / exist[i]
                acc[i + 1] = acc_dist(dist, thr=1)

        sum_dist[0] = cal_ave(exist, sum_dist[1:])
        acc[0] = cal_ave(exist, acc[1:])

        return acc, sum_dist, exist, (preds_maxval.squeeze(dim=2).t(), if_exist), (preds, gt)

    def update(self, acc, dist, exists, maxval, gt, loss):
        self.accLogger.update(acc[0].item(), self.batch_size)
        self.lossLogger.update(loss.item(), self.batch_size)
        self.distLogger.update(dist[0].item(), self.batch_size)
        self.curveLogger.update(maxval.reshape(1, -1).squeeze(), gt.reshape(1, -1).squeeze())

        exists = exists.tolist()
        for k, v in self.pts_acc_Loggers.items():
            self.pts_curve_Loggers[k].update(maxval[k], gt[k])
            if exists[k] > 0:
                self.pts_acc_Loggers[k].update(acc.tolist()[k + 1], exists[k])
                self.pts_dist_Loggers[k].update(dist.tolist()[k + 1], exists[k])

    def get_batch_result(self):
        self.loss, self.acc, self.dist, self.auc, self.pr = self.lossLogger.avg, self.accLogger.avg * 100, \
                 self.distLogger.avg, self.curveLogger.cal_AUC(), self.curveLogger.cal_PR()
        return self.loss, self.acc, self.dist, self.auc, self.pr

    def update_tb(self, tb, iter):
        tb.add_scalar('{}/Loss'.format(self.phase), self.loss, iter)
        tb.add_scalar('{}/Acc'.format(self.phase), self.acc, iter)
        tb.add_scalar('{}/Dist'.format(self.phase), self.dist, iter)
        tb.add_scalar('{}/AUC'.format(self.phase), self.auc, iter)
        tb.add_scalar('{}/PR'.format(self.phase), self.pr, iter)

    def get_kps_result(self):
        body_part_acc = [Logger.avg for k, Logger in self.pts_acc_Loggers.items()]
        body_part_dist = [Logger.avg for k, Logger in self.pts_dist_Loggers.items()]
        body_part_auc = [Logger.cal_AUC() for k, Logger in self.pts_curve_Loggers.items()]
        body_part_pr = [Logger.cal_PR() for k, Logger in self.pts_curve_Loggers.items()]
        return body_part_acc, body_part_dist, body_part_auc, body_part_pr


class EpochEvaluator:
    def __init__(self, out_size):
        self.height, self.width = out_size
        self.kps, self.gts, self.valids = [], [], []
        # self.cal_pckh = PCKHCalculator()

    def update(self, kp, gt, valid):
        self.kps += kp
        self.gts += gt
        self.valids += valid

    def eval_per_epoch(self):
        pckh_ls = self.eval_pckh()
        return [round(pckh, 4) for pckh in pckh_ls]
        # pck = self.eval_pck()
        # return pckh_ls, pck

    def eval_pck(self):
        return 0

    def eval_pckh(self, refp=0.5):
        parts_valid = sum(self.valids)[-12:].tolist()
        parts_correct, pckh = [0] * 12, []
        for i in range(len(self.gts)):
            central = (self.gts[i][-11] + self.gts[i][-12]) / 2
            head_size = np.linalg.norm(np.subtract(central, self.gts[i][0]))
            if not head_size:
                continue
            # valid = np.array(self.valids[i][-12:])
            sum_valid = sum(self.valids[i][-12:]).tolist()
            valid = np.array(list(map(lambda x:2*x-1, self.valids[i][-12:])))
            dist = np.linalg.norm(self.kps[i][-12:] - self.gts[i][-12:], axis=1)
            ratio = dist / head_size
            scale = ratio * valid
            correct_num = sum((0 <= scale) & (scale <= refp))  # valid_joints(a)
            pckh.append(correct_num / sum_valid) if sum_valid > 0 else pckh.append(0)

            for idx, (s, v) in enumerate(zip(scale, valid)):
                if v == 1 and s <= refp:
                    parts_correct[idx] += 1

        parts_pckh = []
        for correct_pt, valid_pt in zip(parts_correct, parts_valid):
            parts_pckh.append(correct_pt / valid_pt) if valid_pt > 0 else parts_pckh.append(0)

        return [sum(pckh) / len(pckh)] + parts_pckh

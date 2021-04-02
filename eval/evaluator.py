from eval.logger import DataLogger, CurveLogger
from eval.utils import *


class BatchEvaluator:
    def __init__(self, kps, phase, bs):
        self.phase = phase
        self.kps = kps
        self.batch_size = bs
        self.accLogger, self.distLogger, self.lossLogger, self.pckhLogger, self.curveLogger = \
            DataLogger(), DataLogger(), DataLogger(), DataLogger(), CurveLogger()
        self.pts_acc_Loggers = {i: DataLogger() for i in range(kps)}
        self.pts_dist_Loggers = {i: DataLogger() for i in range(kps)}
        self.pts_curve_Loggers = {i: CurveLogger() for i in range(kps)}
        self.pts_pckh_Loggers = {i: DataLogger() for i in range(12)}

    def eval_per_batch(self, output, label, out_height):
        label, output = label.cpu().data, output.cpu().data
        preds, preds_maxval = getPreds(output)
        gt, _ = getPreds(label)

        if_exist = torch.Tensor([torch.sum((label[i][j] > 0).float()) > 0 for i in range(len(label))
                                 for j in range(len(label[0]))]).view(len(label), len(label[0])).t()

        norm = torch.ones(preds.size(0)) * out_height / 10
        dists = calc_dists(preds, gt, norm)
        acc, sum_dist, exist = torch.zeros(self.kps + 1), torch.zeros(self.kps + 1), torch.zeros(self.kps)
        pckh = cal_pckh(gt, preds, if_exist.t(), refp=0.5)

        for i, kps_dist in enumerate(dists):
            nums = exist_id(if_exist[i])
            exist[i] = len(nums)
            if len(nums) > 0:
                dist = kps_dist[nums]
                sum_dist[i + 1] = torch.sum(dist) / exist[i]
                acc[i + 1] = acc_dist(dist - 1)

        sum_dist[0] = cal_ave(exist, sum_dist[1:])
        acc[0] = cal_ave(exist, acc[1:])

        return acc, sum_dist, exist, pckh, (preds_maxval.squeeze(dim=2).t(), if_exist)

    def update(self, acc, dist, exists, pckh, maxval, gt, loss):
        self.accLogger.update(acc[0].item(), self.batch_size)
        self.lossLogger.update(loss.item(), self.batch_size)
        self.distLogger.update(dist[0].item(), self.batch_size)
        self.pckhLogger.update(pckh[0], self.batch_size)
        self.curveLogger.update(maxval.reshape(1, -1).squeeze(), gt.reshape(1, -1).squeeze())

        exists = exists.tolist()
        for k, v in self.pts_acc_Loggers.items():
            self.pts_curve_Loggers[k].update(maxval[k], gt[k])
            if exists[k] > 0:
                self.pts_acc_Loggers[k].update(acc.tolist()[k + 1], exists[k])
                self.pts_dist_Loggers[k].update(dist.tolist()[k + 1], exists[k])
        pckh_exist = exists[-12:]
        for k, v in self.pts_pckh_Loggers.items():
            if exists[k] > 0:
                self.pts_pckh_Loggers[k].update(pckh[k + 1], pckh_exist[k])

    def get_batch_result(self):
        self.loss, self.acc, self.pckh, self.dist, self.auc, self.pr = self.lossLogger.avg, self.accLogger.avg * 100, \
                self.pckhLogger.avg * 100, self.distLogger.avg, self.curveLogger.cal_AUC(), self.curveLogger.cal_PR()
        return self.loss, self.acc, self.pckh, self.dist, self.auc, self.pr

    def update_tb(self, tb, iter):
        tb.add_scalar('{}/Loss'.format(self.phase), self.loss, iter)
        tb.add_scalar('{}/Acc'.format(self.phase), self.acc, iter)
        tb.add_scalar('{}/Dist'.format(self.phase), self.dist, iter)
        tb.add_scalar('{}/pckh'.format(self.phase), self.pckh, iter)
        tb.add_scalar('{}/AUC'.format(self.phase), self.auc, iter)
        tb.add_scalar('{}/PR'.format(self.phase), self.pr, iter)

    def get_kps_result(self):
        body_part_acc = [Logger.avg for k, Logger in self.pts_acc_Loggers.items()]
        body_part_dist = [Logger.avg for k, Logger in self.pts_dist_Loggers.items()]
        body_part_auc = [Logger.cal_AUC() for k, Logger in self.pts_curve_Loggers.items()]
        body_part_pr = [Logger.cal_PR() for k, Logger in self.pts_curve_Loggers.items()]
        body_part_pckh = [Logger.avg for k, Logger in self.pts_pckh_Loggers.items()]
        return body_part_acc, body_part_dist, body_part_auc, body_part_pr, body_part_pckh


class EpochEvaluator:
    def __init__(self):
        pass








from sklearn import metrics
import numpy as np
import torch


class DataLogger(object):
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


class CurveLogger:
    def __init__(self):
        self.clear()

    def clear(self):
        self.gt = []
        self.preds = []

    def update(self, gt, preds):
        if len(self.gt) == 0:
            self.gt = gt
            self.preds = preds
        else:
            if len(gt.shape) == 0:
                gt = gt.unsqueeze(dim=0)
                preds = preds.unsqueeze(dim=0)
            self.gt = torch.cat((self.gt, gt))
            self.preds = torch.cat((self.preds, preds))

    def cal_AUC(self):
        try:
            auc = metrics.roc_auc_score(self.preds, self.gt)
        except:
            auc = 0
        return auc

    def cal_PR(self):
        try:
            P, R, thresh = metrics.precision_recall_curve(self.preds, self.gt)
            area = 0
            for idx in range(len(thresh)-1):
                a = (R[idx] - R[idx+1]) * (P[idx+1] + P[idx])/2
                area += a
            return area
        except:
            return 0

    def get_thresh(self):
        try:
            P, R, thresh = metrics.precision_recall_curve(self.preds, self.gt)
            PR_ls = [P[idx] + R[idx] for idx in range(len(P))]
            max_idx = PR_ls.index(max(PR_ls))
            return thresh[max_idx]
        except:
            return 0


class ErrorLogger:
    pass


# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
from scipy.io import savemat, loadmat

from src.opt import opt
from sklearn import metrics
import numpy as np
import os

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict

from utils.img import transformBoxInvert


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


class NullWriter(object):
    def write(self, arg):
        pass


def exist_id(p):
    exist = (p == 0).float()
    ids = []
    for i, item in enumerate(exist):
        if torch.sum(item) < 1:
            ids.append(i)
    return ids


def cal_ave(weights, inps):
    res = 0
    for w, i in zip(weights, inps):
        res += w*i
    return res/torch.sum(weights)

def cal_pckh(y_pred, y_true,if_exist,refp=0.5):
    num_samples = len(y_true)
    name_value = {}
    ls, rs, le, re, lw, rw, lh, rh, lk, rk, la, ra, pckh = [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(num_samples):
        central = (y_true[i][-11] + y_true[i][-12])/2
        head_size = np.linalg.norm(np.subtract(central,y_true[i][0]))
    # for coco datasets, abandon eyes and ears keypoints
        used_joints = range(5,17)
        dist = np.zeros((num_samples, len(used_joints)))
        valid = np.zeros((num_samples, len(used_joints)))
        valid[i] = if_exist[i][5:17]
        dist[i] = np.linalg.norm(y_true[i][5:17] - y_pred[i][5:17],axis=1) / head_size
        jnt_count = valid_joints(if_exist[i][5:17])
        scale = dist * valid
        a = (0 < scale[i, :]) & (scale[i, :] <= refp)
        less_than_threshold = valid_joints(a)
        # acc_num +=
        joint_radio = 100.0 * less_than_threshold / jnt_count
        # PCKh = np.ma.array(scale, mask=False)
        # name_value[i] = [('lS', 100*PCKh[i][0]),
        #                  ('RS',100*PCKh[i][1]),
        #                  ('LE',100*PCKh[i][2]),
        #                  ('RE',100*PCKh[i][3]),
        #                  ('LW',100*PCKh[i][4]),
        #                  ('RW',100*PCKh[i][5]),
        #                  ('LH',100*PCKh[i][6]),
        #                  ('RH',100*PCKh[i][7]),
        #                  ('LK',100*PCKh[i][8]),
        #                  ('RK',100*PCKh[i][9]),
        #                  ('LA',100*PCKh[i][10]),
        #                  ('RA',100*PCKh[i][11]),
        #                  ('PCKh', joint_radio)]
        # name_value = OrderedDict(name_value)
        name = list(scale)
        ls.append(name[i][0])
        rs.append(name[i][1])
        le.append(name[i][2])
        re.append(name[i][3])
        lw.append(name[i][4])
        rw.append(name[i][5])
        lh.append(name[i][6])
        rh.append(name[i][7])
        lk.append(name[i][8])
        rk.append(name[i][9])
        la.append(name[i][10])
        ra.append(name[i][11])
        pckh.append(joint_radio)

    b = cal_average(ls)
    c = cal_average(rs)
    d = cal_average(le)
    e = cal_average(re)
    f = cal_average(lw)
    g = cal_average(rw)
    h = cal_average(lh)
    x = cal_average(rh)
    y = cal_average(lk)
    z = cal_average(rk)
    p = cal_average(la)
    q = cal_average(ra)
    PCkh = sum(pckh) / len(pckh)
    # PCKH = [('lS', b ),
    #         ('RS', c ),
    #         ('LE', d ),
    #         ('RE', e ),
    #         ('LW', f ),
    #         ('RW', g ),
    #         ('LH', h ),
    #         ('RH', x ),
    #         ('LK', y ),
    #         ('RK', z ),
    #         ('LA', p ),
    #         ('RA', q ),
    #         ('PCKh', PCkh)]
    # PCKH = OrderedDict(PCKH)
    PCKH = [PCkh, b, c, d, e, f, g, h, x, y, z,  p, q]
    return PCKH


def cal_pckh2(y_pred, y_true, if_exist, refp=0.5):
    parts_valid = sum(if_exist).tolist()
    parts_correct, pckh = [0]*12, []
    for i in range(len(y_true)):
        central = (y_true[i][-11] + y_true[i][-12]) / 2
        head_size = np.linalg.norm(np.subtract(central, y_true[i][0]))
        valid = np.array(if_exist[i][5:17])
        dist = np.linalg.norm(y_true[i][5:17] - y_pred[i][5:17],axis=1)
        ratio = dist/ head_size
        scale = ratio * valid
        correct_num = sum((0 < scale) & (scale <= refp))#valid_joints(a)
        pckh.append(100.0 * correct_num / sum(valid))

        for idx, (s, v) in enumerate(zip(scale, valid)):
            if v == 1 and s <= refp:
                parts_correct[idx] += 1

    parts_pckh = []
    for correct, valid in zip(parts_correct, parts_valid):
        parts_pckh.append(correct/ valid) if valid > 0 else parts_pckh.append(0)

    # parts_pckh = [correct/ valid for correct, valid in zip(parts_correct, parts_valid)]
    return [sum(pckh)/len(pckh)] + parts_pckh


def cal_average(a):
    count =0
    num = 0
    for i in range(len(a)):
        if a[i] !=0:
            num = num+a[i]
            count +=1
        else:
            num = num
            count += 0
    return num/count

def valid_joints(if_exist):
    count = 0
    for i in range(len(if_exist)):
        if if_exist[i] == 0:
            count += 0
        else:
            count += 1
    return count

def cal_accuracy(output, label, idxs):
    label, output = label.cpu().data, output.cpu().data
    preds, preds_maxval = getPreds(output)
    gt, _ = getPreds(label)

    if_exist = torch.Tensor([torch.sum((label[i][j] > 0).float()) > 0 for i in range(len(label))
                             for j in range(len(label[0]))]).view(len(label),len(label[0])).t()

    norm = torch.ones(preds.size(0)) * opt.outputResH / 10
    dists = calc_dists(preds, gt, norm)
    acc, sum_dist, exist = torch.zeros(len(idxs) + 1), torch.zeros(len(idxs) + 1), torch.zeros(len(idxs))
    # pckh = cal_pckh(gt,preds,if_exist.t(),refp=0.5)
    pckh = cal_pckh2(gt,preds,if_exist.t(),refp=0.5)
    # pckh = cal_everage(name_value)

    for i, kps_dist in enumerate(dists):
        nums = exist_id(if_exist[i])
        exist[i] = len(nums)
        if len(nums) > 0:
            dist = kps_dist[nums]
            sum_dist[i + 1] = torch.sum(dist)/exist[i]
            acc[i + 1] = acc_dist(dist-1)

    sum_dist[0] = cal_ave(exist, sum_dist[1:])
    acc[0] = cal_ave(exist, acc[1:])
    return acc, sum_dist, exist, pckh, (preds_maxval.squeeze(dim=2).t(), if_exist)


def acc_dist(dists, thr=0.5):
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).float().sum() * 1.0 / dists.ne(-1).float().sum()
    else:
        return -1


def accuracy(output, label, dataset, part, out_offset=None):
    # exist = if_exist(part)
    if type(output) == list:
        return accuracy(output[opt.nStack - 1], label[opt.nStack - 1], dataset, part, out_offset)
    else:
        return heatmapAccuracy(output.cpu().data, label.cpu().data, dataset.accIdxs, part)


def part_accuracy(output, label, idx, exist, out_offset=None):
    return heatmapAccuracy(output.cpu().data, label.cpu().data, idx, exist)


def heatmapAccuracy(output, label, idxs, parts):
    preds = getPreds(output)
    gt = getPreds(label)
    norm = torch.ones(preds.size(0)) * opt.outputResH / 10
    dists = calc_dists(preds, gt, norm)


    acc, sum_dist = torch.zeros(len(idxs) + 1), torch.zeros(len(idxs) + 1)
    exists = []
    avg_acc, sum_dist = 0, 0
    cnt = 0
    for i in range(len(idxs)):
        # acc[i + 1] = dist_acc(dists[idxs[i] - 1])
        acc[i + 1], exist = dist_acc(dists[i] - 1, parts[i])
        exists.append(exist)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1
    if cnt != 0:
        acc[0] = avg_acc / cnt
    return acc, exists


def getPreds(hm):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert hm.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hm.view(hm.size(0), hm.size(1), -1), 2)

    maxval = maxval.view(hm.size(0), hm.size(1), 1)
    idx = idx.view(hm.size(0), hm.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hm.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hm.size(3))

    # pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    # preds *= pred_mask
    return preds, maxval

def calc_dists(preds, target, normalize):
    preds = preds.float().clone()
    target = target.float().clone()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] > 0 and target[n, c, 1] > 0:
                dists[c, n] = torch.dist(
                    preds[n, c, :], target[n, c, :]) / normalize[n]
            else:
                dists[c, n] = 0
    return dists


def dist_acc(dists, part, thr=0.5):
    ids = exist_id(part)
    dists = dists[ids]
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).float().sum() * 1.0 / dists.ne(-1).float().sum(), len(ids)
    else:
        return -1, 0


def postprocess(output):
    p = getPreds(output)

    for i in range(p.size(0)):
        for j in range(p.size(1)):
            hm = output[i][j]
            pX, pY = int(round(p[i][j][0])), int(round(p[i][j][1]))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                p[i][j] += diff.sign() * 0.25
    p -= 0.5

    return p


def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(
                round(float(preds[i][j][1])))
            if 1 < pX < opt.outputResW - 2 and 1 < pY < opt.outputResH - 2:
                diff = torch.Tensor(
                    (hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                diff = diff.sign() * 0.25
                diff[1] = diff[1] * inpH / inpW
                preds[i][j] += diff

    preds_tf = torch.zeros(preds.size())
    for i in range(hms.size(0)):        # Number of samples
        for j in range(hms.size(1)):    # Number of output heatmaps for one sample
            preds_tf[i][j] = transformBoxInvert(
                preds[i][j], pt1[i], pt2[i], inpH, inpW, resH, resW)

    return preds, preds_tf, maxval


def getmap(JsonDir='./val/alphapose-results.json'):
    ListDir = '../data/coco/coco-minival500_images.txt'

    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[2]  # specify type here
    prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
    print('Running evaluation for *%s* results.' % (annType))

    # load Ground_truth
    dataType = 'val2014'
    annFile = '../data/coco/%s_%s.json' % (prefix, dataType)
    cocoGt = COCO(annFile)

    # load Answer(json)
    resFile = JsonDir
    cocoDt = cocoGt.loadRes(resFile)

    # load List
    fin = open(ListDir, 'r')
    imgIds_str = fin.readline()
    if imgIds_str[-1] == '\n':
        imgIds_str = imgIds_str[:-1]
    imgIds_str = imgIds_str.split(',')

    imgIds = []
    for x in imgIds_str:
        imgIds.append(int(x))


    # running evaluation
    iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
    t = np.where(0.5 == iouThrs)[0]

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()

    score = cocoEval.eval['precision'][:, :, :, 0, :]
    mApAll, mAp5 = 0.01, 0.01
    if len(score[score > -1]) != 0:
        score2 = score[t]
        mApAll = np.mean(score[score > -1])
        mAp5 = np.mean(score2[score2 > -1])
    cocoEval.summarize()
    return mApAll, mAp5
# def cal_pckh(y_true, y_pred, refp=0.5):
#     for j in range(len(y_true[0])):
#         head_size = (y_true[j][1]+y_true[j][2])/2 - y_true[j][0]
#         assert len(y_true) == len(head_size)
#         num_samples = len(y_true)
#         used_joints = range(4, 16)
#         y_true = y_true[:, used_joints, :]
#         y_pred = y_pred[:, used_joints, :]
#         dist = np.zeros((num_samples, len(used_joints)))
#         valid = np.zeros((num_samples, len(used_joints)))
#
#         for i in range(num_samples):
#             valid[i,:] = valid_joints(y_true[i])
#             dist[i,:] = norm(y_true[i] - y_pred[i], axis=1) / head_size[i]
#         match = (dist <= refp) * valid
#
#         return match.sum() / valid.sum()
#
# def norm(x, axis=None):
#     return np.sqrt(np.sum(np.power(x, 2), axis=axis))
#
# def valid_joints(y, min_valid=0):
#     def and_all(x):
#         if x.all():
#             return 1
#         return 0
#
#     return np.apply_along_axis(and_all, axis=1, arr=(y > min_valid))
#
# if __name__ == '__main__':
#     preds = torch.Tensor([[1,1], [2,2]])
#     gt = torch.Tensor([[1,0], [3,1]])
#     cal_pckh(preds, gt)

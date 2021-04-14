import numpy as np
import torch


def cal_ave(weights, inps):
    res = 0
    for w, i in zip(weights, inps):
        res += w*i
    return res/torch.sum(weights)


def cal_pckh(y_pred, y_true, if_exist, refp=0.5):
    parts_valid = sum(if_exist)[-12:].tolist()
    parts_correct, pckh = [0]*12, []
    for i in range(len(y_true)):
        central = (y_true[i][-11] + y_true[i][-12]) / 2
        head_size = np.linalg.norm(np.subtract(central, y_true[i][0]))
        valid = np.array(if_exist[i][-12:])
        dist = np.linalg.norm(y_true[i][-12:] - y_pred[i][-12:], axis=1)
        ratio = dist/ head_size
        scale = ratio * valid
        correct_num = sum((0 < scale) & (scale <= refp))#valid_joints(a)
        pckh.append(correct_num / sum(valid)) if sum(valid) > 0 else pckh.append(0)

        for idx, (s, v) in enumerate(zip(scale, valid)):
            if v == 1 and s <= refp:
                parts_correct[idx] += 1

    parts_pckh = []
    for correct_pt, valid_pt in zip(parts_correct, parts_valid):
        parts_pckh.append(correct_pt/ valid_pt) if valid_pt > 0 else parts_pckh.append(0)

    # parts_pckh = [correct/ valid for correct, valid in zip(parts_correct, parts_valid)]
    return [sum(pckh)/len(pckh)] + parts_pckh


def acc_dist(dists, thr=0.5):
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).float().sum() * 1.0 / dists.ne(-1).float().sum()
    else:
        return -1


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


def exist_id(p):
    exist = (p == 0).float()
    ids = []
    for i, item in enumerate(exist):
        if torch.sum(item) < 1:
            ids.append(i)
    return ids

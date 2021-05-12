#-*-coding:utf-8-*-
import torch


def xywh2xyxy(box):
    y_max = box[1] + box[3]
    x_max = box[0] + box[2]
    return [box[0], box[1], x_max, y_max]


def kps_reshape(raw_kps):
    kps_num = int(len(raw_kps) / 3)
    kps, kps_valid = [], []
    for i in range(kps_num):
        kps.append([raw_kps[i*3], raw_kps[i*3+1]])
        kps_valid.append(raw_kps[i*3+2])
    return kps, kps_valid


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


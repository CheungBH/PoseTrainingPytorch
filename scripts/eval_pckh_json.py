import os
import numpy as np
import eval_helpers

def computeDist(preFrames, gtFrames):
    distAll = {}
    dist = {}
    keypoints_gt,label = eval_helpers.eval1(gtFrames)
    keypoints_pre = eval_helpers.eval(preFrames)
    for imgidx in keypoints_gt.keys():
        gt = keypoints_gt[imgidx]
        pre = keypoints_pre[imgidx]
        head = gt[0]
        neck = ((gt[1][0] + gt[2][0]) / 2,
                (gt[1][1] + gt[2][1]) / 2)
        label_gt = label[imgidx]
        if imgidx in keypoints_pre.keys():
            for index in range(len(label_gt)):
                if label_gt[index] == 0:
                    dist[index] = 0
                else:
                    pointGT = gt[index]
                    pointPre = pre[index]
                    d = np.linalg.norm(np.subtract(pointGT, pointPre))
                    headSize = eval_helpers.getHeadSize(head[0], neck[0], head[1], neck[1])
                    dnormal = d / headSize * 0.1
                    dist[index] = dnormal
            distAll[imgidx] = dist
            dist = {}
    return distAll

def computePCKh(distAll, distThresh):
    pckh_dist = {}
    idxs = 0
    result = open("test_result.csv", "a")
    result.write(
        "folder_name,image_name,model,pckh_head,pckh_lShoulder,pckh_rShoulder,pckh_lElbow, pckh_rElbow,pckh_rWrist,pckh_rWrist,pckh_lHip, pckh_rHip,pckh_lKnee,pckh_rKnee,pckh_lAnkle,pckh_rAnkle,PCKH\n")
    result.close()
    for id in distAll.keys():
        pckAll = np.zeros([len(distAll[id]) + 1, 1])
        nCorrect = 0
        nTotal = 13
        for pidx in range(len(distAll[id])):
            if distAll[id][pidx] <= distThresh:
                idxs += 1
            if ((distAll[id][pidx]) > 1) | (distAll[id][pidx] == 0):
                distAll[id][pidx] = 1
            pck = 100.0 * (1-distAll[id][pidx])
            pckAll[pidx] = pck
            nCorrect = idxs
        pckAll[-1] = 100.0 * nCorrect / nTotal
        idxs = 0
        pckh_dist[id] = pckAll
        result = open("test_result.csv", "a+")
        result.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                     format(folder_name, id,preFramesAll ,  pckAll[0], pckAll[1], pckAll[2], pckAll[3],
                            pckAll[4], pckAll[5], pckAll[6], pckAll[7], pckAll[8], pckAll[9], pckAll[10], pckAll[11], pckAll[12],pckAll[13]))
    head,ls, rs, le, re, lw, rw, lh, rh, lk, rk, la, ra, pckh = [],[], [], [], [], [], [], [], [], [], [], [], [], []
    for idx in pckh_dist.keys():
        head.append(pckh_dist[idx][0])
        ls.append(pckh_dist[idx][1])
        rs.append(pckh_dist[idx][2])
        le.append(pckh_dist[idx][3])
        re.append(pckh_dist[idx][4])
        lw.append(pckh_dist[idx][5])
        rw.append(pckh_dist[idx][6])
        lh.append(pckh_dist[idx][7])
        rh.append(pckh_dist[idx][8])
        lk.append(pckh_dist[idx][9])
        rk.append(pckh_dist[idx][10])
        la.append(pckh_dist[idx][11])
        ra.append(pckh_dist[idx][12])
        pckh.append(pckh_dist[idx][13])
    a = cal_average(head)
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
    PCkh = cal_average(pckh)
    result = open("test_result.csv", "a+")
    result.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                 format(folder_name, sum, preFramesAll,a,b, c, d, e,f, g, h, x, y, z, p, q,PCkh))
    result.close()

def cal_average(a):
    count = 0
    num = 0
    for i in range(len(a)):
        if a[i] != 0:
            num = num + a[i]
            count += 1
        else:
            num = num
            count += 0
    return num / count

def evaluatePCKh(gtFramesAll, prFramesAll):
    distThresh = 0.5
    distAll = computeDist(gtFramesAll, prFramesAll)
    pckh_dist = computePCKh(distAll, distThresh)
    return pckh_dist

if __name__ == '__main__':
    sum = 'sum'
    folder_name = 'yoga4phone'
    path_name = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/testdata/yoga4/'
    files = os.listdir(path_name)
    gtFramesAll = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/EVAL_JSON/GT/yoga4phone_gt.json"
    for file in files:
        preFramesAll = os.path.join(path_name+file)
        evaluatePCKh(preFramesAll, gtFramesAll)
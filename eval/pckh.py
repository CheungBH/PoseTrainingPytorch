import numpy as np


class PCKHCalculator:
    def __init__(self):
        self.dist = []

    def compute_dis(self,kps,gt,valid):
        for i in range(len(kps)):
            head = gt[0]
            neck = ((gt[1][0] + gt[2][0]) / 2,
                    (gt[1][1] + gt[2][1]) / 2)
            for index in range(len(valid)):
                if valid[index] == 0:
                    self.dist[index] = 0
                else:
                    pointGT = gt[index]
                    pointPre = kps[index]
                    d = np.linalg.norm(np.subtract(pointGT, pointPre))
                    headSize = self.getHeadSize(head[0], neck[0], head[1], neck[1])
                    dnormal = d / headSize * 0.1
                    self.dist[index] = dnormal
        return self.dist

    def getHeadSize(sefl,x1,y1,x2,y2):
        headSize = np.linalg.norm(np.subtract([x2, y2], [x1, y1]))
        if headSize == 0:
            pass
        else:
            return headSize

    def computePCKh(self,distAll, distThresh):
        idxs = 0
        for id in distAll.keys():
            pckAll = np.zeros([len(distAll[id]) + 1, 1])
            nCorrect = 0
            nTotal = 13
            for pidx in range(len(distAll[id])):
                if distAll[id][pidx] <= distThresh:
                    idxs += 1
                if ((distAll[id][pidx]) > 1) | (distAll[id][pidx] == 0):
                    distAll[id][pidx] = 1
                pck = 100.0 * (1 - distAll[id][pidx])
                pckAll[pidx] = pck
                nCorrect = idxs
            pckAll[-1] = 100.0 * nCorrect / nTotal
            idxs = 0
        return pckAll

if __name__ == '__main__':
    dis = PCKHCalculator.compute_dis()
    pckh = PCKHCalculator.computePCKh(dis, 0.5)

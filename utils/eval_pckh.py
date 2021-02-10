import torch
import numpy as np
from collections import OrderedDict

def cal_pckh(y_pred, y_true,if_exist,refp=0.5):
    central=y_true[:,-11,:]+y_true[:,-12,:]
    head_size = norm(central - y_true[:,0,:], axis=1)
    assert len(y_true) == len(head_size)
    num_samples = len(y_true)
    # for coco datasets, abandon eyes and ears keypoints
    used_joints = range(4,16)
    y_true = y_true[:, used_joints,:]
    y_pred = y_pred[:, used_joints,:]
    dist = np.zeros((num_samples,len(used_joints)))
    valid = np.zeros((num_samples,len(used_joints)))
    joint_radio =[]

    for i in range(num_samples):
        valid[i,:] = if_exist[i,:]
        dist[i,:] = norm(y_true[i] - y_pred[i], axis=1) / head_size[i]
        jnt_count = valid_joints(if_exist[i,:])
        scale = dist * valid
        a = (0<scale[i,:])& (scale[i,:] <= refp)
        less_than_threshold = valid_joints(a)
        joint_radio.append(less_than_threshold/jnt_count)

    scale = dist * valid
    PCKh = np.ma.array(scale, mask=False)
    for i in range(num_samples):
        name_value = [('Shoulder', 0.5 * (PCKh[i][0] + PCKh[i][1])),
                      ('Elbow', 0.5 * (PCKh[i][2] + PCKh[i][3])),
                      ('Wrist', 0.5 * (PCKh[i][4] + PCKh[i][5])),
                      ('Hip', 0.5 * (PCKh[i][6] + PCKh[i][7])),
                      ('Knee', 0.5 * (PCKh[i][8] + PCKh[i][9])),
                      ('Ankle', 0.5 * (PCKh[i][10] + PCKh[i][11])),
                      ('PCKh', np.mean(PCKh[i][:])),
                      ('pckh@0.5',np.sum(PCKh[i][:]*joint_radio[i]/3))]
        name_value = OrderedDict(name_value)
        print(name_value)
    return name_value


def norm(x, axis=None):
    return np.sqrt(np.sum(np.power(x, 2).numpy(), axis=axis))

def valid_joints(if_exist):
    count = 0
    for i in range(len(if_exist)):
        if if_exist[i] == 0:
            count += 0
        else:
            count += 1
    return count

if __name__ == '__main__':
    preds = torch.Tensor(([
        [[26., 39.],
         [35., 55.],
         [26., 35.],
         [34., 34.],
         [29., 30.],
         [21., 48.],
         [36., 47.],
         [40., 35.],
         [37., 46.],
         [29., 39.],
         [28., 54.],
         [32., 41.],
         [26., 47.],
         [33., 45.],
         [32., 37.],
         [28., 40.],
         [32., 51.]],

        [[46., 65.],
         [23., 35.],
         [55., 34.],
         [15., 47.],
         [54., 53.],
         [25., 48.],
         [20., 42.],
         [29., 24.],
         [46., 42.],
         [11., 45.],
         [43., 64.],
         [37., 38.],
         [27., 46.],
         [13., 61.],
         [33., 45.],
         [36., 50.],
         [18., 39.]]]))
    gt = torch.Tensor(([
        [[15., 40.],
         [13., 39.],
         [ 0.,  0.],
         [13., 35.],
         [ 0.,  0.],
         [18., 32.],
         [18., 30.],
         [16., 44.],
         [18., 42.],
         [15., 54.],
         [19., 47.],
         [37., 34.],
         [37., 33.],
         [38., 52.],
         [38., 48.],
         [51., 52.],
         [50., 48.]],

        [[12., 27.],
         [12., 26.],
         [ 0.,  0.],
         [16., 26.],
         [ 0.,  0.],
         [18., 32.],
         [18., 31.],
         [16., 43.],
         [16., 42.],
         [14., 52.],
         [13., 50.],
         [34., 40.],
         [32., 40.],
         [35., 53.],
         [34., 51.],
         [48., 54.],
         [46., 52.]]]))
    if_exist = torch.Tensor(([
          [1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1.],
          [1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0.]]))
    # if_exist = if_exist.t()
    cal_pckh(preds, gt, if_exist,refp=0.5)
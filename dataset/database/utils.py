

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


def select_kps(kps, kps_valid, idx_list, all_kps_num):
    new_kps, new_kps_valid = [], []
    for i in range(all_kps_num):
        if i+1 in idx_list:
            new_kps.append(kps[i])
            new_kps_valid.append(kps_valid[i])
    return new_kps, new_kps_valid



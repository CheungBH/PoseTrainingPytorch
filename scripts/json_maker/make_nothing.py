import json
import os

transform = list(zip(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13,14,15,16]
    ))

def load_json(json_file):
    anno = json.load(open(json_file))
    keypoint = {}
    images = []
    bbox = []
    ids = []
    # result = open("test_result.csv", "a")
    # result.write("image_name,x,y,w,h,ids,nose,le,re,lear,rear,ls, rs, le, re, lw, rw, lh, rh, lk, rk, la, ra\n")
    # result.close()
    for img_info in anno['annotations']:
        images.append(img_info['image_name'])
        xs = img_info['keypoints'][0::3]
        ys = img_info['keypoints'][1::3]
        ids = img_info["id"]
        bbox = img_info['bbox']
        new_kp = []
        for idx, idy in transform:
            new_kp.append(
                (xs[idx], ys[idy])
            )
        keypoint[img_info['image_name']] = img_info['keypoints']
    return images, keypoint, bbox, ids


def load_jsons(json_list):
    images, keypoints, bboxes, ids = [], [], [], []
    for json_file in json_list:
        img, kps, box, i = load_json(json_file)
        images += img
        keypoints += kps
        bboxes += box
        ids += i
    return images, keypoints, bboxes, ids


if __name__ == '__main__':
    jsons = ["yoga4phone.json", "yoga_eval.json"]
    a, b, c, d = load_jsons(jsons)
    r = 1

    # preFramesAll = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/test_json/'
    # files = os.listdir(preFramesAll)
    # for file in files:
    #     preFrames = os.path.join(preFramesAll +"/"+file)
    #     eval1(preFrames)
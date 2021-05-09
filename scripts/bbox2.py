import json
from random import random
from PIL import Image
import os


def crop_a(anno):
    for img_info in anno['annotations']:
        fn = img_info["image_name"]
        file_name, type = os.path.splitext(fn)
        new_fn = os.path.join(file_name + "_" + str(img_info["id"]) + ".jpg")
        cropmake(fn, new_fn, img_info["bbox"])


def cropmake(fn,new_fn,bbox):
    loc = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/testdata/yoga4phone/'
    im = Image.open(loc+fn)
    scale_factor = 0.2
    scaleRate = random.uniform(scale_factor)
    bbox = scale(im, bbox, scaleRate)
    x, y, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    bbox = (x, y, x+width, y+height)
    bbox = tuple(bbox)
    newim = im.crop(bbox)
    newloc = os.path.join(image_path, "newImage/")
    newim.save(newloc + str(new_fn))
    im.close()


def scale(im, bbox, rate):
    left, top, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
    imght = im.shape[1]
    imgwidth = im.shape[2]
    x = max(0, left - width * rate / 2)
    y = max(0, top - height * rate / 2)
    bottomRightx = min(imgwidth - 1, left + width * (1+rate / 2))
    bottomRighty = min(imght - 1, top + height * (1+rate / 2))
    return [x, y, bottomRightx, bottomRighty]


if __name__ == '__main__':
    path = '/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/test_json/'
    files = os.listdir(path)
    image_path = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/testdata/"
    for file in files:
        file = os.path.join(path, file)
        fp = open(file, 'r')
        images = json.load(fp)
        fp.close()
        crop_a(images)






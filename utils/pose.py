# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

from utils.img import (load_image, drawGaussian, cropBox, transformBox, cv_rotate)
import torch
import numpy as np
import random
from config.opt import opt

# import config.config as config

inputResH = opt.inputResH
inputResW = opt.inputResW
outputResH = opt.outputResH
outputResW = opt.outputResW


def rnd(x):
    return max(-2 * x, min(2 * x, np.random.randn(1)[0] * x))


def generateSampleBox(img_path, bndbox, part, nJoints, imgset, scale_factor, dataset, train=True):

    img = load_image(img_path)
    if train:
        img[0].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        img[1].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)
        img[2].mul_(random.uniform(0.7, 1.3)).clamp_(0, 1)

    img[0].add_(-0.406)
    img[1].add_(-0.457)
    img[2].add_(-0.480)

    upLeft = torch.Tensor((int(bndbox[0][0]), int(bndbox[0][1])))
    bottomRight = torch.Tensor((int(bndbox[0][2]), int(bndbox[0][3])))
    ht = bottomRight[1] - upLeft[1]
    width = bottomRight[0] - upLeft[0]
    imght = img.shape[1]
    imgwidth = img.shape[2]
    scaleRate = random.uniform(*scale_factor)

    upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
    upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
    bottomRight[0] = min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2)
    bottomRight[1] = min(imght - 1, bottomRight[1] + ht * scaleRate / 2)

    # Doing Random Sample
    if opt.addDPG:
        PatchScale = random.uniform(0, 1)
        if PatchScale > 0.85:
            ratio = ht / width
            if (width < ht):
                patchWidth = PatchScale * width
                patchHt = patchWidth * ratio
            else:
                patchHt = PatchScale * ht
                patchWidth = patchHt / ratio

            xmin = upLeft[0] + random.uniform(0, 1) * (width - patchWidth)
            ymin = upLeft[1] + random.uniform(0, 1) * (ht - patchHt)
            xmax = xmin + patchWidth + 1
            ymax = ymin + patchHt + 1
        else:
            xmin = max(
                1, min(upLeft[0] + np.random.normal(-0.0142, 0.1158) * width, imgwidth - 3))
            ymin = max(
                1, min(upLeft[1] + np.random.normal(0.0043, 0.068) * ht, imght - 3))
            xmax = min(max(
                xmin + 2, bottomRight[0] + np.random.normal(0.0154, 0.1337) * width), imgwidth - 3)
            ymax = min(
                max(ymin + 2, bottomRight[1] + np.random.normal(-0.0013, 0.0711) * ht), imght - 3)

        upLeft[0] = xmin
        upLeft[1] = ymin
        bottomRight[0] = xmax
        bottomRight[1] = ymax

    # Counting Joints number
    jointNum = 0
    # if imgset == 'coco':
    try:
        for i in range(nJoints):
            if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
                    and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
                jointNum += 1
    except ValueError:
        print(part)
        print(part[i][0])
        print(part[i][1])

    # Doing Random Crop
    if opt.addDPG:
        if jointNum > nJoints * 0.8 and train:
            switch = random.uniform(0, 1)
            if switch > 0.96:
                bottomRight[0] = (upLeft[0] + bottomRight[0]) / 2
                bottomRight[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.92:
                upLeft[0] = (upLeft[0] + bottomRight[0]) / 2
                bottomRight[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.88:
                upLeft[1] = (upLeft[1] + bottomRight[1]) / 2
                bottomRight[0] = (upLeft[0] + bottomRight[0]) / 2
            elif switch > 0.84:
                upLeft[0] = (upLeft[0] + bottomRight[0]) / 2
                upLeft[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.80:
                bottomRight[0] = (upLeft[0] + bottomRight[0]) / 2
            elif switch > 0.76:
                upLeft[0] = (upLeft[0] + bottomRight[0]) / 2
            elif switch > 0.72:
                bottomRight[1] = (upLeft[1] + bottomRight[1]) / 2
            elif switch > 0.68:
                upLeft[1] = (upLeft[1] + bottomRight[1]) / 2


    inp = cropBox(img, upLeft, bottomRight, inputResH, inputResW)

    if jointNum == 0:
        inp = torch.zeros(3, inputResH, inputResW)

    # out = torch.zeros(nJoints, outputResH, outputResW)
    # setMask = torch.zeros(nJoints, outputResH, outputResW)
    # # Draw Label
    # # if imgset == 'coco':
    # for i in range(nJoints):
    #     if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
    #        and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
    #         hm_part = transformBox(
    #             part[i], upLeft, bottomRight, inputResH, inputResW, outputResH, outputResW)
    #
    #         out[i] = drawGaussian(out[i], hm_part, opt.hmGauss)
    #
    #     setMask[i].add_(1)
    out, setMask = draw_label(nJoints, part, bottomRight, upLeft)

    #     # Flip
    #     if random.uniform(0, 1) < 0.5:
    #         inp = flip(inp)
    #         out = shuffleLR(flip(out), dataset)
    if train:
        # Rotate
        r = rnd(opt.rotate)
        if random.uniform(0, 1) < 0.6:
            r = 0
        if r != 0:
            inp = cv_rotate(inp, r, inputResW, inputResH)
            out = cv_rotate(out, r, outputResW, outputResH)

    return inp, out, setMask, upLeft, bottomRight


def choose_kps(array, target):
    new_parts = [item for i, item in enumerate(array) if i+1 in target]
    return new_parts


def draw_label(nJoints, part, bottomRight, upLeft):
    out = torch.zeros(nJoints, outputResH, outputResW)
    setMask = torch.zeros(nJoints, outputResH, outputResW)
    # Draw Label
    # if imgset == 'coco':
    for i in range(nJoints):
        if part[i][0] > 0 and part[i][0] > upLeft[0] and part[i][1] > upLeft[1] \
           and part[i][0] < bottomRight[0] and part[i][1] < bottomRight[1]:
            hm_part = transformBox(
                part[i], upLeft, bottomRight, inputResH, inputResW, outputResH, outputResW)

            out[i] = drawGaussian(out[i], hm_part, opt.hmGauss)

        setMask[i].add_(1)
    return out, setMask


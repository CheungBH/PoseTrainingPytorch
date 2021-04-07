#-*-coding:utf-8-*-
import h5py

h5_path = "../data/ceiling/0605_new.h5"
with h5py.File(h5_path, 'r') as annot:
    imgname_coco_train = annot['imgname'][:-10]  #:-5887
    bndbox_coco_train = annot['bndbox'][:-10]
    part_coco_train = annot['part'][:-10]
    # val
    imgname_coco_val = annot['imgname'][-10:]  # -5887:
    bndbox_coco_val = annot['bndbox'][-10:]
    part_coco_val = annot['part'][-10:]

a = 1



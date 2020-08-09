import csv    #加载csv包便于读取csv文件
import os
from train_result.config import task_folder, batch_folder

include_cuda = True

csv_name = "{}.csv".format(os.path.join(task_folder, batch_folder, batch_folder))
out_name = csv_name[:-4] + ".txt"
csv_file = open(csv_name)    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件

data = [line for line in csv_reader_lines]
opt = [item for item in data[0]]

if include_cuda:
    begin = "'CUDA_VISIBLE_DEVICES= python train_opt.py "
else:
    begin = "'python train_opt.py "


def change_name(name):
    if name == "TRUE":
        return 'True'
    elif name == "FALSE":
        return "False"
    else:
        return name


cmds = []
for idx, mdl in enumerate(data[1:]):
    tmp = ""
    valid = False
    for o, m in zip(opt, mdl):
        if m != "":
            tmp += "--"
            tmp += o
            tmp += " "
            tmp += change_name(m)
            tmp += " "
            valid = True

    tmp += "--expFolder {}-{} ".format(task_folder, batch_folder)
    tmp += "--expID {}".format(idx+1)
    cmd = begin + tmp + "'\n"
    if valid:
        cmds.append(cmd)
    else:
        cmds.append("\n")

with open(out_name, "a+") as out:
    for c in cmds:
        out.write(c)

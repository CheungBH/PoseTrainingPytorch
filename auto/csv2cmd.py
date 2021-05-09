import csv    #加载csv包便于读取csv文件
import os
from auto.config import task_folder, excel_name, base_name

include_cuda = True
negative = ["False", "FALSE", "false"]
positive = ["True", "TRUE", "true"]


def csvTransform(file):
    csv_name = file
    out_name = os.path.join(task_folder, "{}.txt".format(base_name))
    csv_file = open(csv_name)    #打开csv文件
    csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件

    data = [line for line in csv_reader_lines]
    opt = [item for item in data[0]]

    if include_cuda:
        begin = "'CUDA_VISIBLE_DEVICES= python trainer.py "
    else:
        begin = "'python trainer.py "

    cmds = []
    for idx, mdl in enumerate(data[1:]):
        tmp = ""
        valid = False
        for o, m in zip(opt, mdl):
            if m in positive:
                tmp += "--"
                tmp += o
                tmp += " "
            elif m in negative:
                continue
            else:
                if m != "":
                    tmp += "--"
                    tmp += o
                    tmp += " "
                    tmp += m
                    tmp += " "
                    valid = True

        tmp += "--expFolder {}-{} ".format(task_folder, base_name)
        tmp += "--expID {}".format(idx+1)
        cmd = begin + tmp + "'\n"
        if valid:
            cmds.append(cmd)
        else:
            cmds.append("\n")

    with open(out_name, "a+") as out:
        for c in cmds:
            out.write(c)


if __name__ == '__main__':
    file_name = os.path.join(task_folder, excel_name)
    csvTransform(file_name)

    # src_folder = "underwater"
    # for batch_name in os.listdir(src_folder):
    #     file_name = "{}.csv".format(os.path.join(src_folder, batch_name, batch_name))
    #     csvTransform(file_name)
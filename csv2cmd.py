import csv    #加载csv包便于读取csv文件

include_cuda = True

csv_file = open('tmp.csv')    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件

data = [line for line in csv_reader_lines]
opt = [item for item in data[0]]

if include_cuda:
    begin = "'CUDA_VISIBLE_DEVICES= python train_opt.py "
else:
    begin = "'python train_opt.py "

cmds = []
for mdl in data[1:]:
    tmp = ""
    for o, m in zip(opt, mdl):
        tmp += o
        tmp += " "
        tmp += m
        tmp += ""
    cmd = begin + tmp + "'"
    cmds.append(cmd)

with open("cmds.txt", "f") as out:
    for c in cmds:
        out.write(c)

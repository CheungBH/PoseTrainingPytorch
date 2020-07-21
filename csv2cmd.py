import csv    #加载csv包便于读取csv文件

include_cuda = True

csv_name = 'training_csv/alphapose_aic_new.csv'
out_name = csv_name[:-4] + ".txt"
csv_file = open(csv_name)    #打开csv文件
csv_reader_lines = csv.reader(csv_file)   #逐行读取csv文件

data = [line for line in csv_reader_lines]
opt = [item for item in data[0]]

if include_cuda:
    begin = "'CUDA_VISIBLE_DEVICES= python train_opt.py "
else:
    begin = "'python train_opt.py "

cmds = []
for idx, mdl in enumerate(data[1:]):
    tmp = ""
    for o, m in zip(opt, mdl):
        if m != "":
            tmp += "--"
            tmp += o
            tmp += " "
            tmp += m
            tmp += " "
    tmp += "--expID {}".format(idx+1)
    cmd = begin + tmp + "'\n"
    cmds.append(cmd)

with open(out_name, "a+") as out:
    for c in cmds:
        out.write(c)

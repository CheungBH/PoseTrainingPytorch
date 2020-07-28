from .config import models_name

with open("{}.txt".format(models_name), "r") as f:
    lines = [line for line in f.readlines()]

train_begin, train_end = 17, 17
CUDA = -1
target_cmds = lines[train_begin-1: train_end]

if CUDA != -1:
    cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in target_cmds]
else:
    cmds = [cmd[0] + cmd[23:-1] + ",\n" for cmd in target_cmds]

with open("tmp.txt", "a+") as cf:
    for cmd in cmds:
        cf.write(cmd)
    cf.write("\n")

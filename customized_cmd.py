with open("training_csv/alphapose_aic.txt", "r") as f:
    lines = [line for line in f.readlines()]

train_begin, train_end = 21, 24
CUDA = -1
target_cmds = lines[train_begin: train_end+1]

if CUDA != -1:
    cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in target_cmds]
else:
    cmds = [cmd[22:-1] + ",\n" for cmd in target_cmds]

with open("tmp.txt", "a+") as cf:
    for cmd in cmds:
        cf.write(cmd)
    cf.write("\n")

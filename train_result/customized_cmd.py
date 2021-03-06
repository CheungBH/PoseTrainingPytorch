import os
from train_result.config import task_folder, batch_folder

CUDA = "0,1,2,3"
train_begin, train_end = 1, 192


def customize(CUDA, file, begin, end):
    with open(file, "r") as f:
        lines = [line for line in f.readlines()]

    target_cmds = lines[begin-1: end]

    if CUDA != -1:
        cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in target_cmds]
    else:
        cmds = [cmd[0] + cmd[23:-1] + ",\n" for cmd in target_cmds]

    with open("{}/cmds.txt".format(os.path.join(task_folder, batch_folder)), "a+") as cf:
        for cmd in cmds:
            cf.write(cmd)
        cf.write("\n")


if __name__ == '__main__':
    file_name = "{}.txt".format(os.path.join(task_folder, batch_folder, batch_folder))
    customize(CUDA, file_name, train_begin, train_end)


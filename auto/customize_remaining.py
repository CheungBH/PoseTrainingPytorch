import os
from .config import task_folder, batch_folder

with open("{}/{}/remaining.txt".format(task_folder, batch_folder), "r") as f:
    lines = [line for line in f.readlines() if line != "\n"]

CUDA = 0

if CUDA != -1:
    cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in lines]
else:
    cmds = [cmd[0] + cmd[23:-1] + ",\n" for cmd in lines]

with open("{}/cmds.txt".format(os.path.join(task_folder, batch_folder)), "w") as cf:
    for cmd in cmds:
        cf.write(cmd)
    cf.write("\n")

from auto.config import task_folder, base_name

CUDA = 0
train_begin, train_end = 1, 17


def customize(CUDA, file, begin, end):
    with open(file, "r") as f:
        lines = [line for line in f.readlines()]

    target_cmds = lines[begin-1: end]

    if CUDA != -1:
        cmds = [cmd[:22] + str(CUDA) + cmd[22:-1] + ",\n" for cmd in target_cmds]
    else:
        cmds = [cmd[0] + cmd[23:-1] + ",\n" for cmd in target_cmds]

    with open("{}/{}_cmds.txt".format(task_folder, base_name), "a+") as cf:
        for cmd in cmds:
            cf.write(cmd)
        cf.write("\n")


if __name__ == '__main__':
    file_name = "{}/{}.txt".format(task_folder, base_name)
    customize(CUDA, file_name, train_begin, train_end)


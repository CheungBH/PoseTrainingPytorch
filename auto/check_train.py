from .config import task_folder, batch_folder, computer
import os

generate_remaining = True

result_path = os.path.join("../result", "{0}-{1}/{0}-{1}_result.csv".format(task_folder, batch_folder))
if not os.path.exists(result_path):
    result_path = os.path.join("../result", "{}-{}_result_{}.csv".format(task_folder, batch_folder, computer))

trained_models = []
with open(result_path, "r") as f:
    for line in f.readlines():
        try:
            trained_models.append(int(line.split(",")[0]))
        except:
            continue
# print(sorted(trained_models))

plan_path = os.path.join(task_folder, batch_folder, "{}.csv".format(batch_folder))
plan_models = []
with open(plan_path, "r") as f:
    for idx, line in enumerate(f.readlines()):
        if len(line) > 30 and "backbone" not in line:
            plan_models.append(idx)
# print(plan_models)

remain_models = [model for model in plan_models if model not in trained_models]
print(remain_models)
wrong_models = [model for model in trained_models if model not in plan_models]
print(wrong_models)

if generate_remaining:
    file_path = os.path.join(task_folder, batch_folder, "remaining.txt")
    commands_path = os.path.join(task_folder, batch_folder, "{}.txt".format(batch_folder))
    with open(commands_path, "r") as f:
        all_cmd = f.readlines()
    with open(file_path, "w") as f:
        for cmd in all_cmd:
            if len(cmd) > 30:
                if int(cmd.split(" ")[-1][:-2]) in remain_models:
                    f.write(cmd)

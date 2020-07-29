from .config import models_name

model_txt = "{0}/{0}.txt".format(models_name)

models = []
with open(model_txt, "r") as f:
    for line in f.readlines():
        try:
            models.append(int(line.split(" ")[-1][:-2]))
        except:
            continue

# print(models)

trained_folder = "{}/result/log_result".format(models_name)

import os
trained = [int(file) for file in os.listdir(trained_folder)]

# print(sorted(trained))
rest = [item for item in models if item not in trained]
print(rest)

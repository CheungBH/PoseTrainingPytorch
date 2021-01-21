import os
import random

# percent = 0.15
src_folder = "../../data/underwater_0612/0612_selected"
files = [f for f in os.listdir(src_folder)]

del_ls = random.sample(files, 162)
print(len(del_ls))

for item in del_ls:
    os.remove(os.path.join(src_folder, item))
    print(item)

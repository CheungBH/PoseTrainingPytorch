import os
import shutil

src_folder = "../../data/0507_0619_0710_2nd_merge/images"
selected_folder = "../../data/ceiling_filtered"

remaining_folder = "../../data/ceiling_remaining"
os.makedirs(remaining_folder, exist_ok=True)

src_files = [f for f in os.listdir(src_folder)]
selected_files = [f for f in os.listdir(selected_folder)]

remain_files = [f for f in src_files if f not in selected_files]

for file in remain_files:
    print(file)
    shutil.copy(os.path.join(src_folder, file), os.path.join(remaining_folder, file))



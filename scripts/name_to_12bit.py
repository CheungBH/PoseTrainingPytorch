import os
import shutil

img_folder = "../data/pose_0612_selected"
dest_folder = img_folder + "_zfill"
os.makedirs(dest_folder, exist_ok=True)
img_names = [i for i in os.listdir(img_folder)]

for idx, pre_name in enumerate(img_names):
    name = str(idx) + ".jpg"
    post_name = name.zfill(15)
    post_name = "3" + post_name
    print(post_name)
    shutil.copy(os.path.join(img_folder, pre_name), os.path.join(dest_folder, post_name))

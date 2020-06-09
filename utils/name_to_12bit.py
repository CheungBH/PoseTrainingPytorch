import os
import shutil

img_folder = "../data/ceiling/surface"
dest_folder = "../data/ceiling/post_surface"
os.makedirs(dest_folder, exist_ok=True)
img_names = [i for i in os.listdir(img_folder)]

for name in img_names:
    post_name = name.zfill(15)
    post_name = "2" + post_name
    print(post_name)
    shutil.copy(os.path.join(img_folder, name), os.path.join(dest_folder, post_name))

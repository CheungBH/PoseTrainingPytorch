import os
import shutil

folder_name = "aic_origin"
src_folder = os.path.join("exp", folder_name)
dest_folder = os.path.join("result", folder_name)
os.makedirs(dest_folder, exist_ok=True)

for folder in os.listdir(src_folder):
    result_folder = os.path.join(src_folder, folder, folder)
    if os.path.exists(os.path.join(result_folder, "loss.jpg")):
        try:
            shutil.copytree(result_folder, os.path.join(dest_folder, folder))
        except FileExistsError:
            pass



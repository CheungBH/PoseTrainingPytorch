import os

src_folder = "../exp/aic_origin"

for folder in os.listdir(src_folder):
    sub_folder = os.path.join(src_folder, folder)
    for file in os.listdir(sub_folder):
        file_path = os.path.join(sub_folder, file)
        if not os.path.isdir(file_path):
            if "best" in file_path or "option" in file_path or "img" in file_path:
                continue
            else:
                os.remove(file_path)

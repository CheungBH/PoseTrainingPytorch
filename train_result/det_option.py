import os

src_folder = "../exp/underwater-seres101_17kps"

for folder in os.listdir(src_folder):
    sub_folder = os.path.join(src_folder, folder)
    for file in os.listdir(sub_folder):
        file_path = os.path.join(sub_folder, file)
        if not os.path.isdir(file_path):
            if "_20" in file_path:
                print(file_path)
                os.remove(file_path)
            else:
                continue

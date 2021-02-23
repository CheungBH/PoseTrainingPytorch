import os

src_folder = 'underwater'

for sub_folder in os.listdir(src_folder):
    sub_path = os.path.join(src_folder, sub_folder)
    for file in os.listdir(sub_path):
        if "txt" in file:
            file_path = os.path.join(sub_path, file)
            os.remove(file_path)
            print(file_path)

import os
import shutil

src_folder = "../onnx/0407"
dest_folder = "../onnx/0407_selected"

onnx_sim = True
libtorch = True

for folder in os.listdir(src_folder):
    os.makedirs(os.path.join(dest_folder, folder))
    sub_folder = os.path.join(src_folder, folder)
    for file in os.listdir(sub_folder):
        if onnx_sim and "sim" in file:
            shutil.copy(os.path.join(sub_folder, file), os.path.join(dest_folder, folder, "model.onnx"))
        elif libtorch and ".pt" in file and ".pth" not in file:
            shutil.copy(os.path.join(sub_folder, file), os.path.join(dest_folder, folder, "model.pt"))
        else:
            continue





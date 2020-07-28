import os
import shutil

src = "aic_origin"

for folder in os.listdir(src):
    shutil.copy(os.path.join(src, folder, "img.jpg"), os.path.join(src, folder, folder, "img.jpg"))

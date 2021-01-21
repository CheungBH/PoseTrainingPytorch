import os
import shutil

src = "../exp/aic_origin"

for folder in os.listdir(src):
    try:
        shutil.copy(os.path.join(src, folder, "img.jpg"), os.path.join(src, folder, folder, "img.jpg"))
    except:
        pass
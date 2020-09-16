import os
import shutil

src_folder = '../exp/ceiling_prune'
dest_folder = "../test_weight/ceiling_prune"
os.makedirs(dest_folder,exist_ok=True)

target_indic = ["acc", "_pr", "auc", "dist"]

for folder in os.listdir(src_folder):
    sub_folder = os.path.join(src_folder, folder)
    sub_dest_folder = os.path.join(dest_folder, folder)
    for file in os.listdir(sub_folder):
        for indic in target_indic:
            if indic in file:
                target = dest_folder+"/{}-".format(folder)+indic
                os.makedirs(target,exist_ok=True)
                shutil.copy(os.path.join(src_folder, "{}/".format(folder), file), os.path.join(target, file))
                shutil.copy(os.path.join(src_folder, "{}/".format(folder), "option.pkl"),
                            os.path.join(target,"option.pkl"))


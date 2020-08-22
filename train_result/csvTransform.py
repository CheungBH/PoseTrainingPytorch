from train_result.customized_cmd import customize
from train_result.csv2cmd import csvTransform
import os
from train_result.config import batch_folder, task_folder

CUDA_dict = {0: [],
             1: [],
             2: [],
             3: [],
             -1: [],
             "0,1,2,3": []}

csv_name = "{}.csv".format(os.path.join(task_folder, batch_folder, batch_folder))
csvTransform(csv_name)
txt_name = "{}.txt".format(os.path.join(task_folder, batch_folder, batch_folder))

for CUDA, boundary in CUDA_dict.items():
    if not boundary:
        continue
    assert len(boundary) == 2 and isinstance(boundary[0], int) and isinstance(boundary[1], int), \
        "Please check the boundary"
    customize(CUDA, txt_name, boundary[0], boundary[1])

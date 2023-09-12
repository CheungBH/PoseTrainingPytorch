import os
from tester import Tester
from utils.test_utils import write_test_title
from config.config import computer
import csv
from utils.utils import init_model_list_with_kw
from dataset.dataloader import TestLoader

keyword = ["latest"]
folder_keyword = []


class AutoTester:
    def __init__(self, model_folder, data_info, batchsize=8, num_worker=0, same_data_cfg=False):
        self.model_folder = model_folder
        self.model_ls, self.model_cfg_ls, self.data_cfg_ls, self.option_ls = \
            init_model_list_with_kw(model_folder, keyword, fkws=folder_keyword)
        if not same_data_cfg:
            self.data_info = data_info
        else:
            self.data_info = TestLoader(data_info, self.data_cfg_ls[0])
        self.test_csv = os.path.join(self.model_folder, "test_{}.csv".format(computer))
        self.tested = os.path.exists(self.test_csv)
        self.batch_size = batchsize
        self.num_worker = num_worker

    def write_result(self):
        with open(self.test_csv, "a+", newline="") as test_file:
            csv_writer = csv.writer(test_file)
            if not self.tested:
                csv_writer.writerow(write_test_title(self.kps))
                self.tested = True
            test_row = [self.model_folder.replace("\\", "/").split("/")[-1], self.model]
            test_row += self.benchmark
            test_row.append(computer)
            test_row += self.performance
            for part in self.parts:
                test_row.append("")
                test_row += part
            test_row.append("")
            test_row += self.thresh
            csv_writer.writerow(test_row)

    def run(self, phase="test"):
        model_nums = len(self.model_ls)
        for idx, (model, model_cfg, data_cfg, option) in enumerate(zip(
                self.model_ls, self.model_cfg_ls, self.data_cfg_ls, self.option_ls)):
            self.model = model
            print("[{}/{}] Processing model: {}".format(idx+1, model_nums, self.model))
            test = Tester(model_cfg, self.model, self.data_info, data_cfg, batchsize=self.batch_size,
                          num_worker=self.num_worker, phase=phase)
            test.test()
            test.get_benchmark()
            self.kps = test.kps
            self.benchmark, self.performance, self.parts, self.thresh = test.summarize()
            self.write_result()


if __name__ == '__main__':
    dataset = "coco"
    model_folder = "weights/players"
    from config.config import datasets_info
    data_info = [{dataset: datasets_info[dataset]}]
    AT = AutoTester(model_folder, data_info)
    AT.run(phase="test")

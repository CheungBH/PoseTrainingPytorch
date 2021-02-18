from dataset.loader import TestDataset
import os
from tester import Tester
from utils.test_utils import write_test_title
from config.config import computer


class AutoTester:
    def __init__(self, model_folder, data_info, separate=False, batchsize=8, num_worker=1):
        self.model_folder = model_folder
        self.keyword = ["acc", "auc", "pr", "dist"]
        self.separate = separate
        self.test_loader = TestDataset(data_info).build_dataloader(batchsize, num_worker)
        self.model_ls = []
        self.test_csv = os.path.join(self.model_folder, "test_{}.csv".format(computer))
        self.test_exist = os.path.exists(self.test_csv)

    def load_model_and_option(self):
        for folder in os.listdir(self.model_folder):
            if "csv" in folder:
                continue

            for file in os.listdir(os.path.join(self.model_folder, folder)):
                for kw in self.keyword:
                    if kw in file:
                        self.model_ls.append(os.path.join(self.model_folder, folder, file))

    def load_model_option_1by1(self):
        for folder in os.listdir(self.model_folder):
            if "csv" in folder:
                continue

            for file in os.listdir(os.path.join(self.model_folder, folder)):
                if "option" not in file and ".pkl" in file or ".pth" in file:
                    model = os.path.join(self.model_folder, folder, file)
                else:
                    continue
            try:
                self.model_ls.append(model)
            except:
                raise FileNotFoundError("Target model doesn't exist!")

    def write_result(self):
        with open(self.test_csv, "a+") as test_file:
            if self.test_exist:
                test_file.write(write_test_title())
            test_row = [self.model_folder.replace("\\", "/").split("/")[-1], self.model]
            test_row += self.benchmark
            test_row.append(computer)
            test_row += self.performance
            for part in self.parts:
                test_row.append("")
                test_row += part
            test_row.append("")
            test_row += self.thresh

    def run(self):
        if self.separate:
            self.load_model_option_1by1()
        else:
            self.load_model_and_option()

        model_nums = len(self.model_ls)
        for idx, self.model in enumerate(self.model_ls):
            print("[{}/{}] Processing model: {}".format(idx+1, model_nums, self.model))
            test = Tester(self.test_loader, self.model, print_info=False)
            test.build_with_opt()
            test.test()
            test.get_benchmark()
            self.benchmark, self.performance, self.parts, self.thresh = test.summarize()
            self.write_result()

            if self.separate:
                test.save_thresh_to_option()


if __name__ == '__main__':
    model_folder = "exp/auto_test"
    data_info = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}
    separate = False
    auto_tester = AutoTester(model_folder, data_info, separate=separate)
    auto_tester.run()

from dataset.loader import TestDataset
import os
from tester import Tester


class AutoTester:
    def __init__(self, model_folder, data_info, separate=False, batchsize=8, num_worker=1):
        self.model_folder = model_folder
        self.keyword = ["acc", "auc", "pr", "dist"]
        self.separate = separate
        self.test_loader = TestDataset(data_info).build_dataloader(batchsize, num_worker)
        self.model_ls = []

    def load_model_and_option(self):
        for folder in os.listdir(self.model_folder):
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

    def run(self):
        if self.separate:
            self.load_model_option_1by1()
        else:
            self.load_model_and_option()

        for model in zip(self.model_ls):
            test = Tester(self.test_loader, model)
            test.build_with_opt()
            test.test()
            test.get_benchmark()
            benchmark, performance, parts, thresh = test.summarize()
            if self.separate:
                test.save_thresh_to_option()


if __name__ == '__main__':
    model_folder = ""
    data_info = {}
    separate = False
    auto_tester = AutoTester(model_folder, data_info, separate=separate)




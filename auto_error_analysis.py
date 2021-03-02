from error_analysis import ErrorAnalyser
import os
from dataset.loader import TestDataset
from collections import defaultdict


class AutoErrorAnalyser:
    def __init__(self, img_folder, data_info, num_worker=1):
        self.img_folder = img_folder
        self.cfg_ls = []
        self.model_ls = []
        self.test_loader = TestDataset(data_info).build_dataloader(1, num_worker)
        self.performance = defaultdict(list)
        self.result_file = os.path.join(img_folder, "analyse_result.csv")

    def load_model(self):
        for folder in os.listdir(self.img_folder):
            if not os.path.isdir(os.path.join(self.img_folder, folder)):
                continue

            sub_folder = os.path.join(self.img_folder, folder)
            cfg = None
            for file in os.listdir(sub_folder):
                if "cfg" in file:
                    cfg = os.path.join(sub_folder, file)
                elif "option" not in file and "pkl" in file:
                    model = os.path.join(sub_folder, file)
                else:
                    continue

            try:
                self.model_ls.append(model)
            except:
                raise FileNotFoundError("Model doesn't exist! Please check")
            self.cfg_ls.append(cfg)

    def write_result(self):
        '''
        sample:
        models---JQK
        imgs---ab
        index---"index1", "index2", "index3", "index4"
        '''

        m = ["J", "Q", "K"]
        performance = {
            "a": [[1,2,4,6], [2,5,6,9], [5,6,7,9]],
            "b": [[5,6,7,8], [6,7,8,9], [3,4,6,8]],
        }

    def analyse(self):
        self.load_model()
        for idx, (cfg, model) in enumerate(zip(self.cfg_ls, self.model_ls)):
            analyser = ErrorAnalyser(self.test_loader, model)
            analyser.build_with_opt()
            analyser.analyse()
            performance = analyser.summarize()
            for img_name, perf in performance.items():
                self.performance[img_name].append(perf)
        self.write_result()


if __name__ == '__main__':
    img_folder = "exp/selected"
    analyse_data = {"ceiling": ["data/ceiling/ceiling_test", "data/ceiling/ceiling_test.h5", 0]}
    AEA = AutoErrorAnalyser(img_folder, analyse_data)
    AEA.analyse()

from error_analysis import ErrorAnalyser
import os
from dataset.dataloader import TestLoader
from collections import defaultdict
from utils.utils import init_model_list
from utils.test_utils import write_test_title
import csv
from config.config import computer


class AutoErrorAnalyser:
    def __init__(self, model_folder, data_info, same_data_cfg=True):
        self.model_folder = model_folder
        self.model_ls, self.model_cfg_ls, self.data_cfg_ls, _ = init_model_list(self.model_folder)
        self.performance = defaultdict(list)
        self.analysis_file = os.path.join(model_folder, "analyse_result.csv")
        self.target_ind = ["loss", "acc", "dist", "valid(default)", "valid(customized)"]
        self.test_csv = os.path.join(self.model_folder, "test_{}.csv".format(computer))
        self.tested = os.path.exists(self.test_csv)
        if same_data_cfg:
            self.test_data = TestLoader(data_info, self.data_cfg_ls[0])
        else:
            self.test_data = data_info

    def write_analysis_result(self):
        # print(self.performance)
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

    def write_test_result(self):
        with open(self.test_csv, "a+", newline="") as test_file:
            csv_writer = csv.writer(test_file)
            if not self.tested:
                csv_writer.writerow(write_test_title(self.kps))
                self.tested = True
            test_row = [self.model_folder.replace("\\", "/").split("/")[-1], self.model_path]
            test_row += self.benchmark
            test_row.append(computer)
            test_row += self.test_performance
            for part in self.parts:
                test_row.append("")
                test_row += part
            test_row.append("")
            test_row += self.thresh
            csv_writer.writerow(test_row)

    def analyse(self):
        model_nums = len(self.model_ls)
        for idx, (model_cfg, model_path, data_cfg) in enumerate(zip(self.model_cfg_ls, self.model_ls, self.data_cfg_ls)):
            print("-------------------[{}/{}]: Begin Analysing {}--------------".format(idx+1, model_nums, model_path))
            self.model_path = model_path
            analyser = ErrorAnalyser(model_cfg, model_path, self.test_data, data_cfg, dataset)
            analyser.analyse()
            analyser.get_benchmark()
            self.kps = analyser.kps

            performance = analyser.summarize()
            self.benchmark, self.test_performance, self.parts, self.thresh = analyser.summarize_test()

            for i in range(len(performance[0])):
                self.performance[(performance[0][i], performance[1][i])].append([performance[2][i], performance[3][i],
                                                                                 performance[4][i], performance[5][i],
                                                                                 performance[6][i]])
            self.write_test_result()
        self.write_analysis_result()


if __name__ == '__main__':
    dataset = "yoga"
    model_folder = r"C:\Users\hkuit164\Downloads\0622"
    from config.config import datasets_info
    data_info = [{dataset: datasets_info[dataset]}]
    AEA = AutoErrorAnalyser(model_folder, data_info)
    AEA.analyse()

from error_analysis import ErrorAnalyser
import os
from dataset.dataloader import TestLoader
from collections import defaultdict
from utils.utils import init_model_list


class AutoErrorAnalyser:
    def __init__(self, model_folder, data_info, same_data_cfg=True):
        self.model_folder = model_folder
        self.model_ls, self.model_cfg_ls, self.data_cfg_ls, _ = init_model_list(self.model_folder)
        self.performance = defaultdict(list)
        self.result_file = os.path.join(model_folder, "analyse_result.csv")
        self.target_ind = ["loss", "acc", "dist", "valid(default)", "valid(customized)"]
        if same_data_cfg:
            self.test_data = TestLoader(data_info, self.data_cfg_ls[0])
        else:
            self.test_data = data_info

    def write_result(self):
        print(self.performance)
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
        model_nums = len(self.model_ls)
        for idx, (model_cfg, model_path, data_cfg) in enumerate(zip(self.model_cfg_ls, self.model_ls, self.data_cfg_ls)):
            print("-------------------[{}/{}]: Begin Analysing {}--------------".format(idx+1, model_nums, model_path))
            analyser = ErrorAnalyser(model_cfg, model_path, self.test_data, data_cfg)
            analyser.analyse()
            performance = analyser.summarize()
            for i in range(len(performance[0])):
                self.performance[(performance[0][i], performance[1][i])].append([performance[2][i], performance[3][i],
                                                                                 performance[4][i], performance[5][i],
                                                                                 performance[6][i]])
        self.write_result()


if __name__ == '__main__':
    dataset = "ceiling"
    model_folder = "exp/error_test"
    from config.config import datasets_info
    data_info = [{dataset: datasets_info[dataset]}]
    AEA = AutoErrorAnalyser(model_folder, data_info)
    AEA.analyse()

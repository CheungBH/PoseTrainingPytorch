from sparse import SparseDetector
import os
import csv
from utils.utils import init_model_list_with_kw

keyword = ["latest"]
folder_keyword = ["13"]


class AutoSparseDetector:
    def __init__(self, model_folder, thresh_range=(50, 99), step=1, methods=["shortcut"]):
        self.folder = model_folder
        self.sparse_results = {}
        self.range = thresh_range
        self.step = step
        self.methods = methods
        self.model_ls, self.model_cfg_ls, _, _ = init_model_list_with_kw(model_folder, keyword, folder_keyword)

    def xlsx_title(self):
        tmp = ["model name"]
        for i in list(range(100))[self.range[0]:self.range[1]:self.step]:
            tmp.append(i)
        return tmp

    def write_xlsx(self):
        with open(self.excel_path, "w", newline="") as test_file:
            csv_writer = csv.writer(test_file)
            csv_writer.writerow(self.xlsx_title())
            for model_name, sparse_info in self.sparse_results.items():
                csv_writer.writerow([model_name] + sparse_info)

    def run(self):
        for method in self.methods:
            print("\n---------------------Detecting {} pruning---------------------".format(method))
            self.excel_path = os.path.join(self.folder, "sparse_{}_result.csv".format(method))
            model_num = len(self.model_ls)
            for idx, (cfg, model) in enumerate(zip(self.model_cfg_ls, self.model_ls)):
                print("[{}/{}] Begin processing model {}".format(idx+1, model_num, model))
                sd = SparseDetector(model, model_cfg=cfg, print_info=False, method=method)
                sd.detect()
                self.sparse_results[model] = sd.get_result_ls()
            self.write_xlsx()


if __name__ == '__main__':
    model_folder = "exp/test_kps"
    methods = ["shortcut", "ordinary"]
    asd = AutoSparseDetector(model_folder, methods=methods)
    asd.run()

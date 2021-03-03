from sparse import SparseDetector
import os
import csv


class AutoSparseDetector:
    def __init__(self, model_folder, model_kw=None, thresh_range=(50, 99), step=1, methods=["shortcut"]):
        self.folder = model_folder
        self.models = []
        self.model_kw = model_kw
        self.sparse_results = {}
        self.range = thresh_range
        self.step = step
        self.methods = methods

    def load_models(self):
        for folder in os.listdir(self.folder):
            if not os.path.isdir(os.path.join(self.folder, folder)):
                continue
            for file in os.listdir(os.path.join(self.folder, folder)):
                file_path = os.path.join(self.folder, folder, file)
                if "option" not in file and "cfg" not in file and "pkl" in file:
                    if self.model_kw:
                        for kw in self.model_kw:
                            if kw in file:
                                self.models.append(file_path)
                    else:
                        self.models.append(file_path)

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
            self.excel_path = os.path.join(self.folder, "sparse_{}_result.csv".format(method))
            self.load_models()
            model_num = len(self.models)
            for idx, model in enumerate(self.models):
                print("---------------------[{}/{}] Begin processing model {}--------------".format(idx+1, model_num, model))
                sd = SparseDetector(model, print_info=False, method=method)
                sd.detect()
                self.sparse_results[model] = sd.get_result_ls()
            self.write_xlsx()


if __name__ == '__main__':
    model_kw = ["acc", "dist", "auc", "pr"]
    model_folder = "exp/auto_test_pckh"
    methods = ["shortcut", "ordinary"]
    asd = AutoSparseDetector(model_folder, model_kw, methods=methods)
    asd.run()


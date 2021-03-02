from sparse import SparseDetector
import os
import csv


class AutoSparseDetector:
    def __init__(self, model_folder, model_kw=None, thresh_range=(50,99), step=1):
        self.folder = model_folder
        self.models = []
        self.model_kw = model_kw
        self.sparse_results = {}
        self.xlsx_path = os.path.join(self.folder, "sparse_result.xlsx")
        self.range = thresh_range
        self.step = step

    def load_models(self):
        for folder in os.listdir(self.folder):
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
        for i in self.range[::self.step]:
            tmp.append(i)
        return tmp

    def write_xlsx(self):
        with open(self.xlsx_path, "a+", newline="") as test_file:
            csv_writer = csv.writer(test_file)
            csv_writer.writerow(self.xlsx_title())
        # for model_name, sparse_info in self.sparse_results.items():
        #     
        #
    def run(self):
        model_num = len(self.models)
        for idx, model in enumerate(self.models):
            print("---------------------[{}/{}] Begin processing model {}--------------".format(idx, model_num, model))
            sd = SparseDetector(model, print_info=False)
            sd.detect()
            self.sparse_results[model] = sd.get_sparse_result()
        self.write_xlsx()



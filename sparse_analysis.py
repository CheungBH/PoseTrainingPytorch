import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


class SparseAnalyser:
    def __init__(self, perf_targets, model_index="model name"):
        self.targets = perf_targets
        self.model_index = model_index
        self.target_dic = defaultdict()

    def run(self, sparse_csv, performance_csv, result_csv):
        df = pd.read_csv(sparse_csv, float_precision='high')
        df_header = df.T.iloc[:1, :]
        df1 = df.T.iloc[1:, :]
        df1.plot()
        df_acc = pd.read_csv(performance_csv)
        result_dic = df_acc.groupby('id')['val_acc'].apply(list).to_dict()
        slopes = df1.diff()
        slopes.fillna(value=0, inplace=True)
        final_id = pd.concat([df_header, slopes.squeeze().idxmax().to_frame().T], axis=0)
        final_id = final_id.T
        final_id = final_id[final_id[0].astype('int') > 90]
        final_id['acc'] = final_id[self.model_index].apply(lambda x: result_dic[x.split('/')[-2]])
        final_id.to_csv(result_csv)
        plt.legend(loc=2)
        plt.show()


if __name__ == "__main__":
    s_csv = 'exp/sparse_shortcut_result.csv'
    acc_csv = 'exp/train_seresnet_sparse-laptop win.csv'
    new_csv = 'exp/new.csv'
    targets = ["val_acc", "val_auc"]
    # generate_thred(sparse_csv,acc_csv,new_csv)
    SA = SparseAnalyser(targets)
    SA.run(s_csv, acc_csv, new_csv)
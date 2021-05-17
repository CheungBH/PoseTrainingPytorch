from utils.utils import csv_body_part
from .train_utils import pckh_title


# def check_option_file(path):
#     model_path = path.replace("\\", "/")
#     option_path = "/".join(model_path.split("/")[:-1]) + "/option.pkl"
#     return option_path


def list_to_str(ls):
    string = ""
    for item in ls:
        string += str(item)
        string += ","
    return string[:-1]


def parse_thresh(thresh):
    thresh = thresh.split(",")
    return [float(item) for item in thresh]


def write_test_title(kps):
    title = ["model ID", "model name", "flops", "params", "inf_time", "location", "test_acc", "test_loss", "test_pckh",
             "test_dist", "test_auc", "test_pr", " "]
    title += pckh_title("test")
    title += csv_body_part("test", "acc", kps)
    title += csv_body_part("test", "dist", kps)
    title += csv_body_part("test", "AUC", kps)
    title += csv_body_part("test", "PR", kps)
    title += csv_body_part("test", "thresh", kps)
    return title


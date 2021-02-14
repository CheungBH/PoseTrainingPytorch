
def check_option_file(path):
    model_path = path.replace("\\", "/")
    option_path = "/".join(model_path.split("/")[:-1]) + "/option.pkl"
    return option_path


def list_to_str(ls):
    string = ""
    for item in ls:
        string += str(item)
        string += ","
    return string[:-1]


def parse_thresh(thresh):
    return thresh.split(",")

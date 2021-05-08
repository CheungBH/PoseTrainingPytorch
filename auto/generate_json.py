import sys

data_cfg_path = "../config/data_cfg/data_default.json"
model_cfg_path = "../config/model_cfg/default/cfg_resnet18.json"


def generate_json():
    args = sys.argv
    dest_data_path, dest_cfg_path = args[2], args[3]


if __name__ == '__main__':
    generate_json()


"python generate_json.py data_cfg1.json model_cfg1.json --kps 13 --backbone seresnet18 --loadModel model.pth --sigma 4"
"python generate_json.py data_cfg2.json model_cfg2.json --LR 1E-3 --backbone seresnet101 --sigma 2 --input_height 256 --input_width 256 --output_height 64 --output_width 64"
"python generate_json.py data_cfg3.json model_cfg3.json --LR 1E-3 --backbone seresnet101 --sigma 1 --input_height 320 --input_width 320 --output_height 80 --output_width 80 --loadModel model.pth --se_ratio -1 --scale 0.3 --optMethod adam"

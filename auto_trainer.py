from trainer import Trainer
from utils.train_utils import generate_cmd
import sys
import os
from config.config import bad_epochs, warm_up, train_info


model_related = []
data_related = []


class AutoTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.cmd = generate_cmd(sys.argv[1:])
        self.command = sys.argv
        self.generate_cfg_file()
        self.print()

    def print(self):
        print("----------------------------------------------------------------------------------------------------")
        print(self.opt)
        print("This is the model with id {}".format(self.opt.expID))
        # print("Training backbone is: {}".format(self.opt.backbone))
        # dataset_str = ""
        # for k, v in train_info.items():
        #     dataset_str += k
        #     dataset_str += ", "
        # print("Training data is: {}".format(dataset_str[:-1]))
        print("Warm up end at {}".format(warm_up))
        for k, v in bad_epochs.items():
            if v > 1:
                raise ValueError("Wrong stopping accuracy!")
        print("----------------------------------------------------------------------------------------------------")

    def generate_cfg_file(self):
        dest_folder = os.path.join("exp", self.opt.expFolder, self.opt.expID)
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        else:
            pass
            # if self.opt.resume:
            #     pass
            # else:
            #     raise FileExistsError("Target folder {} exists".format(dest_folder))
        dest_data_cfg = os.path.join(dest_folder, "data_cfg.json")
        dest_model_cfg = os.path.join(dest_folder, "model_cfg.json")
        cfg_cmd = "python auto/generate_json.py {} {} ".format(dest_data_cfg, dest_model_cfg) + self.cmd
        os.system(cfg_cmd)
        self.opt.data_cfg, self.opt.model_cfg = dest_data_cfg, dest_model_cfg

    def train(self):
        trainer = Trainer(self.opt)
        trainer.process()


if __name__ == '__main__':
    from config.opt import opt
    AT = AutoTrainer(opt)
    AT.train()




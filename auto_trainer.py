from trainer import Trainer
from utils.train_utils import generate_cmd
import sys
from config.config import bad_epochs, warm_up, train_info


class AutoTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.cmd = generate_cmd(sys.argv[1:])
        self.unify_cmd()
        self.print()

    def print(self):
        print("----------------------------------------------------------------------------------------------------")
        print(self.opt)
        print("This is the model with id {}".format(self.opt.expID))
        # print("Training backbone is: {}".format(self.opt.backbone))
        dataset_str = ""
        for k, v in train_info.items():
            dataset_str += k
            dataset_str += ", "
        print("Training data is: {}".format(dataset_str[:-1]))
        print("Warm up end at {}".format(warm_up))
        for k, v in bad_epochs.items():
            if v > 1:
                raise ValueError("Wrong stopping accuracy!")
        print("----------------------------------------------------------------------------------------------------")

    def unify_cmd(self):
        if "--freeze_bn False" in self.cmd:
            self.opt.freeze_bn = False
        if "--addDPG False" in self.cmd:
            self.opt.addDPG = False

    def train(self):
        trainer = Trainer(self.opt)
        trainer.process()


if __name__ == '__main__':
    from config.opt import opt
    AutoTrainer(opt).train()




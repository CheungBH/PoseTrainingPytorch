import os
# from utils.utils import
from trainer import Trainer

class AutoTrainer:
    def __init__(self, opt):
        from config.config import train_info
        self.train_info = train_info
        self.opt = opt
        expFolder, expID = opt.expFolder, opt.expID
        dest_folder = os.path.join("exp", expFolder, expID)
        # self.final_results =



    def collect_results(self):


    def process(self, opt):
        self.trainer = Trainer(opt)
        try:
            self.trainer.process()
        except





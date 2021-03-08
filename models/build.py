from .utils.utils import parse_cfg
from torch import nn

class ModelBuilder:
    def __init__(self, cfg_file):
        self.cfg = parse_cfg(cfg_file)
        self.model = self.build()

    def build(self):
        backbone = self.build_backbone()
        head = self.build_head()
        return nn.Sequential([backbone, head])

    def build_backbone(self):


    def build_head(self):


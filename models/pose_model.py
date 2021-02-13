
class PoseModel:
    def __init__(self):
        pass
        
    def build(self, backbone, cfg):
        if backbone == "mobilenet":
            from models.mobilenet.MobilePose import createModel
            from config.model_cfg import mobile_opt as model_ls
        elif backbone == "seresnet101":
            from models.seresnet.FastPose import createModel
            from config.model_cfg import seresnet_cfg as model_ls
        elif backbone == "efficientnet":
            from models.efficientnet.EfficientPose import createModel
            from config.model_cfg import efficientnet_cfg as model_ls
        elif backbone == "shufflenet":
            from models.shufflenet.ShufflePose import createModel
            from config.model_cfg import shufflenet_cfg as model_ls
        elif backbone == "seresnet18":
            from models.seresnet18.FastPose import createModel
            from config.model_cfg import seresnet18_cfg as model_ls
        else:
            raise ValueError("Your model name is wrong")

        model_cfg = model_ls[cfg]
        self.model = createModel(model_cfg)


    def load(self, model_path):
        pass

    def freeze(self):
        pass

    def freeze_bn(self):
        pass

    def init_with_opt(self, opt):





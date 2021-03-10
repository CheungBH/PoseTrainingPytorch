from models.pose_model import PoseModel
import torch

model_path = "../buffer/shortcut_seresnet101.pth"
model_cfg = "../buffer/cfg_shortcut_seresnet101.json"

posenet = PoseModel(device="cpu")
posenet.build(model_cfg)
posenet.load(model_path)
model = posenet.model
print(model)

y = model(torch.randn(1, 3, 320, 256))



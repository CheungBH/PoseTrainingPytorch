from models.pose_model import PoseModel
import torch

model_path = "../buffer/layer_seresnet50.pth"
model_cfg = "../buffer/cfg_layer_seresnet50.json"

posenet = PoseModel(device="cpu")
posenet.build(model_cfg)
posenet.load(model_path)
model = posenet.model
print(model)

y = model(torch.randn(1, 3, 320, 256))



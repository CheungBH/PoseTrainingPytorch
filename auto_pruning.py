import os
from channel_layer_prune import ChannelLayerPruner
from utils.utils import get_corresponding_cfg


model_path = "exp/test_kps/aic_13/latest.pth"
model_cfg = ""
data_cfg = ""
dataset_name = "thermal"


if not model_path or not data_cfg:
    model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])


prune_configs = [
    [80, 2],
    [75, 3],
    [75, 2]
]

model_dest = "pruned_models"

for prune_config in prune_configs:
    assert len(prune_config) == 2, "Wrong pruning config values. Please check"
    dest_folder = os.path.join(model_dest, "Channel-{}_Layer-{}".format(prune_config[0], prune_config[1]))
    dest_cfg, dest_weight = os.path.join(dest_folder, "cfg.json"), os.path.join(dest_folder, "model.pth")
    prune = ChannelLayerPruner(model_path, model_cfg, dest_weight, dest_cfg)
    prune.run(*prune_config)
    results = prune.test(data_cfg, dataset_name)




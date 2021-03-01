from torch import nn

def obtain_prune_idx_50(model):
    all_bn_id, normal_idx, head_idx, shortcut_idx, downsample_idx = [], [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "seresnet50" in layer[0]:
                if "downsample" in layer[0]:
                    downsample_idx.append(i)
                elif "bn1" in layer[0] or "bn2" in layer[0] and i > 5:
                    normal_idx.append(i)
                elif "bn3" in layer[0]:
                    shortcut_idx.append(i)
                else:
                    print("???????")
            else:
                head_idx.append(i)
    return all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx

def obtain_prune_idx2(model):
    all_bn_id, normal_idx, head_idx, shortcut_idx, downsample_idx = [], [], [], [], []
    for i, layer in enumerate(list(model.named_modules())):
        if isinstance(layer[1], nn.BatchNorm2d):
            all_bn_id.append(i)
            if "seresnet18" in layer[0]:
                if i < 5:
                    shortcut_idx.append(i)
                elif "downsample" in layer[0]:
                    downsample_idx.append(i)
                elif "bn1" in layer[0] and i > 5:
                    normal_idx.append(i)
                elif "bn3" in layer[0]:
                    shortcut_idx.append(i)
                else:
                    print("???????")
            else:
                head_idx.append(i)
    return all_bn_id, normal_idx, shortcut_idx, downsample_idx, head_idx

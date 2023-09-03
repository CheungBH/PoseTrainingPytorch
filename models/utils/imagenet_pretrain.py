import torch


def load_pretrain(src_dict, backbone):
    if backbone == "seresnet18":
        target_dict = torch.load("../weights/pretrain/resnet18.pth")
        return load_pretrain_seresnet18(src_dict, target_dict)
    elif backbone == "mobilenet":
        target_dict = torch.load("../weights/pretrain/mobilenet.pth")
        return load_pretrain_mobilenet(src_dict, target_dict)
    elif backbone == "shufflenet":
        target_dict = torch.load("../weights/pretrain/shufflenet.pth")
        return load_pretrain_shufflenet(src_dict, target_dict)
    elif backbone == "seresnet50":
        target_dict = torch.load("../weights/pretrain/resnet50.pth")
        return load_pretrain_seresnet50(src_dict, target_dict)
    elif backbone == "seresnet101":
        return
    else:
        print("Current backbone doesn't support loading imagenet pretrain model")
        return src_dict


def load_pretrain_seresnet18(src, target):
    for idx, (name, param) in enumerate(target.items()):
        if "fc" in name:
            continue
        if "conv2" in name:
            name = name.replace("conv2", "conv3")
        elif "bn2" in name:
            name = name.replace("bn2", "bn3")
        else:
            name = name
        name = "backbone." + name
        src[name] = param
    return src


def load_pretrain_mobilenet(src, target):
    for idx, (name, param) in enumerate(target.items()):
        if "classifier" in name:
            continue
        name = "backbone." + name
        src[name] = param
    return src


def load_pretrain_shufflenet(src, target):
    for idx, (name, param) in enumerate(target.items()):
        if "fc" in name:
            continue
        name = "backbone." + name
        src[name] = param
    return src


def load_pretrain_seresnet50(src, target):
    for idx, (name, param) in enumerate(target.items()):
        if "fc" in name:
            continue

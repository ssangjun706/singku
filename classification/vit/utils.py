import torch.nn as nn


def init_weight(model):
    cname = model.__class__.__name__
    if cname.find("Linear") != -1:
        nn.init.kaiming_normal_(model.weight)

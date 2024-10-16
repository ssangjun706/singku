import torch


def normalize(data):
    mean = torch.mean(data)
    std = torch.std(data)
    return (data - mean) / std


def iou_metric(pred, y):
    pred = torch.argmax(pred, dim=1)
    y = torch.argmax(y, dim=1)
    intersect = torch.logical_and(y, pred).sum().item()
    union = torch.logical_or(y, pred).sum().item()
    return intersect / union

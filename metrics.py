import torch

def accuracy(logits, label):
    total = label.shape[0]
    _, idx = torch.max(logits, dim=1)
    corrects = torch.sum(torch.eq(idx, label))
    return corrects/total
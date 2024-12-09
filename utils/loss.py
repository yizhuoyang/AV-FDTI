import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from random import *
import torch

def _neg_loss(pred, gt):
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1 - epsilon)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    num_pos = pos_inds.float().sum()

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 1) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 1) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

import torch
import torch.nn.functional as F
# from numpy import dot
# from numpy.linalg import norm
from torch.nn import CosineSimilarity

cos = CosineSimilarity(dim=1)

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    # loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = (1 - cos(x, y)).pow_(alpha)  #$ x, y as inputs already normalized??

    loss = loss.mean()
    return loss


def sig_loss(x, y):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss
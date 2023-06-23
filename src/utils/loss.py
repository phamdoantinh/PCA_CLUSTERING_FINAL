import math
import torch
import torch.nn as nn


class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()

    def forward(self, A, B):
        errA = torch.zeros(A.shape[0])
        for i in range(A.shape[0]):
            errA[i] = torch.linalg.norm((A[i] - B[i]))
        return errA.mean()

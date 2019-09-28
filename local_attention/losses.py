from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss']


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, device=None):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(self.device, dtype=torch.float)
        targets.required_grad = False
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, local_inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (FFnum_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        local_features_dist_ap, local_features_dist_an = [], []
        for i in range(n):
            p_pos = -1
            p_max = 0
            n_pos = -1
            n_min = 0
            for j in range(n):
                if i == j:
                    continue
                if mask[i][j] == 0 and (n_pos == -1 or n_min > dist[i][j]):
                    n_pos = j
                    n_min = dist[i][j]
                if mask[i][j] == 1 and (p_pos == -1 or p_max < dist[i][j]):
                    p_pos = j
                    p_max = dist[i][j]
            tmp = torch.dot(local_inputs[i], local_inputs[i]) + torch.dot(local_inputs[p_pos], local_inputs[p_pos])
            tmp += -2 * torch.dot(local_inputs[i], local_inputs[p_pos])
            local_features_dist_ap.append(tmp.clamp(min=1e-12).sqrt().unsqueeze(0))

            tmp = torch.dot(local_inputs[i], local_inputs[i]) + torch.dot(local_inputs[n_pos], local_inputs[n_pos])
            tmp += -2 * torch.dot(local_inputs[i], local_inputs[n_pos])
            local_features_dist_an.append(tmp.clamp(min=1e-12).sqrt().unsqueeze(0))

            dist_ap.append(dist[i][p_pos].unsqueeze(0))
            dist_an.append(dist[i][n_pos].unsqueeze(0))

        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        local_features_dist_ap = torch.cat(local_features_dist_ap)
        local_features_dist_an = torch.cat(local_features_dist_an)

        # print(local_features_dist_an.size())
        # print(local_features_dist_ap.size())
        # print(dist_ap.size())
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        # y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        local_features_loss = self.ranking_loss(local_features_dist_an, local_features_dist_ap, y)
        return loss, local_features_loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        classes = Variable(classes)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


if __name__ == '__main__':
    pass
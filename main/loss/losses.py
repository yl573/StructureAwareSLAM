import torch
from main.data.record_loader import NUM_PLANES
import torch.nn.functional as F
import numpy as np


def calcAssignment(W):
    """
    Takes in the score between planes and calculates the assignment that maximizes the overall sore
    :param W: score matrix, [b, i, j] is the score between pred plane i and gt plane j for batch b
    """
    W = np.array(W)
    numOwners = int(W.shape[0])
    numGoods = int(W.shape[1])
    P = np.zeros(numGoods)
    O = np.full(shape=(numGoods,), fill_value=-1)
    delta = 1.0 / (numGoods + 1)
    queue = list(range(numOwners))
    while len(queue) > 0:
        ownerIndex = queue[0]
        queue = queue[1:]
        weights = W[ownerIndex]
        goodIndex = (weights - P).argmax()
        if weights[goodIndex] >= P[goodIndex]:
            if O[goodIndex] >= 0:
                queue.append(O[goodIndex])
                pass
            O[goodIndex] = ownerIndex
            P[goodIndex] += delta
            pass
        continue
    return O


def find_plane_assignment(planes_pred, planes_gt):
    score_matrix = -torch.abs(planes_pred.unsqueeze(1) - planes_gt.unsqueeze(2))
    score_matrix_norm = score_matrix.norm(dim=3).detach()
    # score_matrix_norm has size (batch, n_planes, n_planes)
    # score_matrix_norm[b, i, j] is the score between pred plane i and gt plane j for batch b

    batch_dim = planes_pred.size(0)
    assignment = []
    for i in range(batch_dim):
        plane_assignment = calcAssignment(score_matrix_norm[i, :, :])
        assignment.append(torch.tensor(plane_assignment))

    assignment = torch.stack(assignment, dim=0)
    return assignment


def permute_planes(planes, assignment):
    """
    Permutes the predicted planes based on the assignment
    :param planes: (batch, num_planes, 3)
    :param assignment: (batch, num_planes), assignment[i,j] is the pred plane corresponding to the jth gt plane
    """
    for i, order in enumerate(assignment):
        planes[i, :] = planes[i, order]
    return planes


def permute_segmentation(seg, assignment):
    for i, order in enumerate(assignment):
        # leave "non-plane" index unchanged
        order = torch.cat((order, torch.tensor([NUM_PLANES])))
        seg[i, :] = seg[i, order]
    return seg


def calc_plane_loss(planes_pred, planes_gt, num_planes):
    # create valid plane mask
    valid_mask = torch.zeros(planes_gt.size())
    for i, n in enumerate(num_planes):
        valid_mask[i, :n, :] = 1

    loss_full = F.mse_loss(planes_pred, planes_gt, reduction='none')
    # mask out the invalid planes
    loss_full = loss_full * valid_mask
    # then take the mean of the loss
    loss_reduced = torch.mean(loss_full)
    return loss_reduced


def calc_seg_loss(seg_pred, seg_gt):
    loss = F.cross_entropy(seg_pred, seg_gt.long())
    return loss

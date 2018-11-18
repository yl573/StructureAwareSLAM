# Ported functions from the original PlaneNet paper
import torch
import numpy as np

MAX_DEPTH = 10


def calcPlaneDepthsModule(width, height, planes, metadata, return_ranges=False):
    """
    Ported from original PlaneNet
    :param width: depth width
    :param height: depth height
    :param planes: (batch, num_planes, 3)
    :param metadata: [focal_x, focal_y, center_x, center_y, camera_width, camera_height]
    :return planeDepths: (batch, height, width, 1)
    """
    urange = (torch.arange(width, dtype=torch.float32).view((1, -1)).repeat(height, 1) / (float(width) + 1) * (
            metadata[4] + 1) - metadata[2]) / metadata[0]
    vrange = (torch.arange(height, dtype=torch.float32).view((-1, 1)).repeat(1, width) / (float(height) + 1) * (
            metadata[5] + 1) - metadata[3]) / metadata[1]
    ranges = torch.stack([urange, torch.ones(urange.shape), -vrange], dim=-1).to(device=planes.device)

    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)

    normalXYZ = torch.sum(ranges.unsqueeze(-2) * planeNormals.unsqueeze(-3).unsqueeze(-3), dim=-1)
    normalXYZ[normalXYZ == 0] = 1e-4

    planeDepths = planeOffsets.squeeze(-1).unsqueeze(-2).unsqueeze(-2) / normalXYZ
    planeDepths = torch.clamp(planeDepths, min=0, max=MAX_DEPTH)
    if return_ranges:
        return planeDepths, ranges
    return planeDepths


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

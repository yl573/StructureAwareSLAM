
import torch
import numpy as np

MAX_DEPTH = 10

def oneHotModule(inp, depth):
    print(inp)
    print(depth)
    inpShape = [int(size) for size in inp.shape]
    inp = inp.view(-1)
    out = torch.zeros(int(inp.shape[0]), depth)
    out.scatter_(1, inp.unsqueeze(-1), 1)
    out = out.view(inpShape + [depth])
    return out

def assignmentModule(W):
    O = calcAssignment(W.detach().cpu().numpy())
    return torch.from_numpy(O)

def calcAssignment(W):
    numOwners = int(W.shape[0])
    numGoods = int(W.shape[1])
    P = np.zeros(numGoods)
    O = np.full(shape=(numGoods, ), fill_value=-1)
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

## The module to compute plane depths from plane parameters
def calcPlaneDepthsModule(width, height, planes, metadata, return_ranges=False):
    urange = (torch.arange(width, dtype=torch.float32).view((1, -1)).repeat(height, 1) / (float(width) + 1) * (
                metadata[4] + 1) - metadata[2]) / metadata[0]
    vrange = (torch.arange(height, dtype=torch.float32).view((-1, 1)).repeat(1, width) / (float(height) + 1) * (
                metadata[5] + 1) - metadata[3]) / metadata[1]
    ranges = torch.stack([urange, torch.ones(urange.shape), -vrange], dim=-1)

    planeOffsets = torch.norm(planes, dim=-1, keepdim=True)
    planeNormals = planes / torch.clamp(planeOffsets, min=1e-4)

    normalXYZ = torch.sum(ranges.unsqueeze(-2) * planeNormals.unsqueeze(-3).unsqueeze(-3), dim=-1)
    normalXYZ[normalXYZ == 0] = 1e-4
    planeDepths = planeOffsets.squeeze(-1).unsqueeze(-2).unsqueeze(-2) / normalXYZ
    planeDepths = torch.clamp(planeDepths, min=0, max=MAX_DEPTH)
    if return_ranges:
        return planeDepths, ranges
    return planeDepths


## The module to compute depth from plane information
def calcDepthModule(width, height, planes, segmentation, non_plane_depth, metadata):
    planeDepths = calcPlaneDepthsModule(width, height, planes, metadata)
    allDepths = torch.cat([planeDepths.transpose(-1, -2).transpose(-2, -3), non_plane_depth], dim=1)
    return torch.sum(allDepths * segmentation, dim=1)
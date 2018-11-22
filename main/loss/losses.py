import torch
from main.data.record_loader import NUM_PLANES
import torch.nn.functional as F
import numpy as np
from main.loss.modules import calcAssignment, calcPlaneDepthsModule
from main.utils import Timer

MAX_DEPTH = 10


def find_plane_assignment(planes_pred, planes_gt):
    """
    Calculates assignment for a batch of predicted planes
    """
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

def find_plane_assignment_from_seg(seg_pred, seg_gt):

    with Timer('plane assignment') as t:

        batches = seg_pred.size(0)
        planes = seg_pred.size(1) - 1
        # get rid of the last dimension for non-planar depth
        seg_pred = seg_pred[:, :planes, :, :]
        match_scores = np.zeros((batches, planes, planes))

        for batch in range(batches):
            for gt_plane in range(planes):
                match_scores[batch, gt_plane]
                gt_plane_seg = seg_gt[batch, :, :] == gt_plane
                for pred_plane in range(planes):
                    pred_plane_seg = seg_pred[batch, pred_plane, :, :]
                    score = torch.sum(gt_plane_seg.float() * pred_plane_seg)
                    match_scores[batch, pred_plane, gt_plane] = score

        assignment = []
        for i in range(batches):
            plane_assignment = calcAssignment(match_scores[i, :, :])
            assignment.append(torch.tensor(plane_assignment))

        assignment = torch.stack(assignment, dim=0)
        return assignment


def permute_planes(planes, assignment):
    """
    Permutes the predicted planes based on the assignment
    :param planes: (batch, num_planes, 3)
    :param assignment: (batch, num_planes), assignment[i,j] is the pred plane corresponding to the jth gt plane
    """
    planes_perm = torch.zeros(planes.size()).to(device=planes.device)
    for i, order in enumerate(assignment):
        planes_perm[i, :] = planes[i, order]
    return planes_perm


def permute_segmentation(seg, assignment):
    seg_perm = torch.zeros(seg.size()).to(device=seg.device)
    for i, order in enumerate(assignment):
        # leave "non-plane" index unchanged
        order = torch.cat((order, torch.tensor([NUM_PLANES])))
        seg_perm[i, :] = seg[i, order]
    return seg_perm


def calc_plane_loss(planes_pred, planes_gt, num_planes):
    # create valid plane mask
    valid_mask = torch.zeros(planes_gt.size()).to(device=planes_pred.device)
    for i, n in enumerate(num_planes):
        valid_mask[i, :n, :] = 1

    loss_full = F.mse_loss(planes_pred, planes_gt, reduction='none')
    # mask out the invalid planes
    loss_full = loss_full * valid_mask
    # then take the mean of the loss
    loss_reduced = torch.mean(loss_full)

    assert loss_reduced.item() > 0, 'invalid plane loss: {}'.format(loss_reduced)
    return loss_reduced


def calc_seg_loss(seg_pred, seg_gt):
    loss = F.cross_entropy(seg_pred, seg_gt.long())
    return loss


def get_metadata(calib, cam_height, cam_width):
    focal_x = calib[0, 0]
    focal_y = calib[1, 1]
    center_x = calib[0, 2]
    center_y = calib[1, 2]
    return torch.tensor([focal_x, focal_y, center_x, center_y, cam_width, cam_height])


def calc_all_depth(depth_pred, planes_pred, seg_pred, calib, cam_height, cam_width):
    width = seg_pred.size(3)
    height = seg_pred.size(2)
    metadata = get_metadata(calib, cam_height, cam_width)
    plane_depth = calcPlaneDepthsModule(width, height, planes_pred, metadata)

    if len(depth_pred.size()) == 3:
        # (batch, W, H) => (batch, 1, W, H)
        depth_pred = depth_pred.unsqueeze(1)

    # concatenate the non-plane depth prediction with the plan depths
    # result: (batch, W, H, NUM_PLANES+1)
    all_depth = torch.cat([plane_depth.permute(0, 3, 1, 2), depth_pred], dim=1)
    depth_pred = torch.sum(all_depth * seg_pred, 1)

    return depth_pred


def calc_depth_loss(all_depth_pred, depth_gt):

    # convert shape from (B, H, W) to (B, 1, H, W)
    depth_mask = ((depth_gt > 1e-4) & (depth_gt < MAX_DEPTH)).float().unsqueeze(1)
    depth_gt = depth_gt.unsqueeze(1)

    depth_error = torch.pow(all_depth_pred - depth_gt, 2)
    # mask out depth too close or too far
    depth_error = depth_error * depth_mask
    depth_loss = depth_error.mean()

    assert depth_loss.item() > 0, 'invalid depth loss: {}'.format(depth_error)

    return depth_loss

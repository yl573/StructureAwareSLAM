import torch
import torch.nn.functional as F
from main.loss.modules import assignmentModule, oneHotModule


def calc_plane_loss(planes_pred, planes_gt):
    loss = 0
    for plane_id in range(planes_pred.size(1)):
        # calculate plane error wrt to each gt plane
        pred_plane = planes_pred[:, plane_id:plane_id + 1, :]  # user id:id+1 to keep dimension
        plane_errors = torch.norm(planes_gt - pred_plane, 2, 2)  # take the norm along the vector dimension

        min_error, min_error_planes = torch.min(plane_errors, 1)  # find the gt plane with the minimum error
        loss += min_error.mean()  # use the minimum error as the loss
    return loss

# def calc_plane_loss(planes_pred, planes_gt, num_planes):
#     distances = torch.norm(planes_gt.unsqueeze(2) - planes_pred.unsqueeze(1), dim=-1)
#
#     print(planes_gt.unsqueeze(2).size())
#     print(planes_pred.unsqueeze(1).size())
#
#     exit()


# W = distances.max() - distances.transpose(1, 2)
# mapping = torch.stack([assignmentModule(W[batchIndex]) for batchIndex in range(len(distances))], dim=0)
# mapping = oneHotModule(mapping.view(-1), depth=int(planes_pred.shape[1])).view((int(mapping.shape[0]), int(mapping.shape[1]), -1))
#
# planes_pred_shuffled = torch.matmul(mapping, planes_pred)
#
# validMask = (torch.arange(int(planes_gt.shape[1]), dtype=torch.int64) < num_planes.unsqueeze(-1)).float()
# plane_loss = torch.sum(F.mse_loss(planes_pred_shuffled, planes_gt) * validMask.unsqueeze(-1)) / torch.sum(validMask)

# return plane_loss

# def segmentation_loss(seg_pred, seg_gt):
#     return F.cross_entropy(depth_pred, depth_gt)

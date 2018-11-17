import torch
import torch.nn.functional as F
from main.loss.modules import assignmentModule, oneHotModule


def calc_plane_loss(planes_pred, planes_gt, num_planes):
    loss = 0
    plane_assoc = torch.zeros(planes_gt.shape[:2], dtype=torch.int) - 1

    # for each image in the batch
    for batch_id in range(len(num_planes)):
        # for each pred plane
        for plane_id in range(num_planes[batch_id].item()):
            # calculate plane error wrt to each pred plane
            gt_plane = planes_gt[batch_id, plane_id:plane_id + 1, :]  # user id:id+1 to keep dimension
            # take the norm along the vector dimension
            plane_errors = torch.norm(planes_pred[batch_id] - gt_plane, 2, 1)
            # find the pred plane with the minimum error
            min_error, min_error_plane = torch.min(plane_errors, 0)
            loss += min_error  # use the minimum error as the loss

            plane_assoc[batch_id, plane_id] = min_error_plane

    return loss, plane_assoc

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

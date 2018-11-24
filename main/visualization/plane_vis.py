import torch
import numpy as np
import torchvision.utils as vutils
from main.utils import Timer


def get_colormap():
    return torch.tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
        [128, 255, 0],
        [255, 128, 0],
        [0, 128, 255],
        [0, 255, 128],
        [255, 0, 128],
        [128, 0, 255],
        [80, 128, 255],
        [255, 80, 128],
        [80, 255, 128],
        [80, 128, 255],
        [255, 230, 180],
        [255, 0, 255],
        [0, 255, 255],
        [100, 0, 0],
        [0, 0, 0],
    ], dtype=torch.float)


def concat_batch_to_height(tensor):
    split_tensors = [tensor[i] for i in range(len(tensor))]
    return torch.cat(split_tensors, 1)


def draw_seg_images(seg):
    colormap = get_colormap()

    # (batch, classes, H, W)
    assignment_shape = (seg.shape[0], colormap.size(0), seg.shape[1], seg.shape[2])

    # (batch, classes, 1, H, W)
    gt_assignment = torch.zeros(assignment_shape).scatter_(1, seg.unsqueeze(1).long(), 1).unsqueeze(2)

    # (1, classes, 3, H, W)
    colormap_expanded = colormap.view(1, colormap.size(0), colormap.size(1), 1, 1).float()

    # (batch, 3, H, W)
    colored_planes, _ = torch.max(gt_assignment * colormap_expanded, dim=1)

    return colored_planes

def mono_to_color(t):
    t = t.unsqueeze(1)
    return torch.cat([t, t, t], dim=1)

def draw_vis(img, pred_seg, gt_seg, all_pred_depth, gt_depth):
    img = img.cpu().detach()
    pred_seg = pred_seg.cpu().detach()
    gt_seg = gt_seg.cpu().detach()
    all_pred_depth = all_pred_depth.cpu().detach()
    gt_depth = gt_depth.cpu().detach()

    with Timer('draw') as t:
        pred_seg_assignment = torch.argmax(pred_seg, dim=1)
        pred_seg = draw_seg_images(pred_seg_assignment).float()

        gt_seg = draw_seg_images(gt_seg).float()

        img = img.permute(0, 3, 1, 2)

        max_depth = gt_depth.max()
        pred_depth_color = mono_to_color(all_pred_depth / max_depth * 255)
        gt_depth_color = mono_to_color(gt_depth / max_depth * 255)

        vis_tensor = torch.cat((img.float(), pred_seg, gt_seg, pred_depth_color, gt_depth_color), dim=0) / 255

        return vutils.make_grid(vis_tensor, nrow=pred_seg.size(0))

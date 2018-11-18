from main.data.record_loader import RecordLoader, NUM_PLANES
import torch
from attrdict import AttrDict


def img_to_torch(img):
    img = (img / 255) - 0.5
    tensor = torch.tensor(img, dtype=torch.float)
    return tensor.permute(0, 3, 1, 2)


# def seg_to_torch_onehot(seg):
#     tensor_shape = [seg.shape[0], NUM_PLANES + 1, seg.shape[1], seg.shape[2]]
#     raw_seg_tensor = torch.tensor(seg, dtype=torch.long).unsqueeze(1) # add extra onehot dimension
#     seg_onehot = torch.zeros(tensor_shape, dtype=torch.float)
#     print(torch.max(raw_seg_tensor))
#     seg_onehot = seg_onehot.scatter_(1, raw_seg_tensor, 1)
#     return seg_onehot


class PlaneNetDataLoader:

    def __init__(self, rec_path, rec_type, batch_size):
        self.rec_loader = RecordLoader(rec_path, rec_type, batch_size)

    def __iter__(self):
        self.rec_loader = iter(self.rec_loader)
        return self

    def __len__(self):
        return len(self.rec_loader)

    def __next__(self):
        raw_batch = next(self.rec_loader)

        batch = AttrDict()
        batch.image_norm = img_to_torch(raw_batch['image_raw'])
        batch.plane = torch.tensor(raw_batch['plane'], dtype=torch.float)
        batch.num_planes = torch.tensor(raw_batch['num_planes'], dtype=torch.int)
        batch.depth = torch.tensor(raw_batch['depth'], dtype=torch.float)
        batch.segmentation_raw = torch.tensor(raw_batch['segmentation_raw'], dtype=torch.int)
        batch.calib = torch.tensor(raw_batch['calib'][0], dtype=torch.float32)

        # common width and height for each batch
        batch.cam_width = raw_batch['extra'][0, 0]
        batch.cam_height = raw_batch['extra'][0, 1]

        return batch

from main.data.record_loader import RecordLoader, NUM_PLANES
import torch


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
        batch = next(self.rec_loader)
        batch['image_norm'] = img_to_torch(batch['image_raw'])
        batch['plane'] = torch.tensor(batch['plane'], dtype=torch.float)
        batch['num_planes'] = torch.tensor(batch['num_planes'], dtype=torch.int)
        batch['depth'] = torch.tensor(batch['depth'], dtype=torch.float)
        batch['segmentation_raw'] = torch.tensor(batch['segmentation_raw'], dtype=torch.int)

        return batch

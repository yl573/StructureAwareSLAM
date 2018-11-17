from main.data.record_loader import RecordLoader
import torch


def img_to_torch(img):
    img = (img / 255) - 0.5
    tensor = torch.tensor(img, dtype=torch.float)
    return tensor.permute(0, 3, 1, 2)


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

        return batch

from main.data.data_loaders import PlaneNetDataLoader
from main.models.planenet import PlaneNet
from main.args import parse_args
import torch

from main.loss.losses import calc_plane_loss

class Trainer:

    def __init__(self, args):
        self.data_loader = PlaneNetDataLoader(args.val_path, 'val', args.batchSize)
        self.model = PlaneNet(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.LR)
        self.device = self.get_device()

    def get_device(self):
        if torch.cuda.is_available():
            return torch.cuda.device(0)
        else:
            return 'cpu'

    def batch_to_device(self, batch):
        for k, v, in batch.items():
            if type(v) == torch.Tensor:
                batch[k] = v.to(device=self.device)
        return batch

    def train(self):
        print('Training on device: '.format(self.device))

        for i in range(10):
            batch = next(self.data_loader)
            batch = self.batch_to_device(batch)
            planes, segmentation, depth = self.model(batch['image_norm'])

            plane_loss = calc_plane_loss(planes, batch['plane'])

            loss = torch.sum(plane_loss)

            print(loss.item())

            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    args = parse_args()
    args.val_path = '/Volumes/MyPassport/planes_scannet_val.tfrecords'
    trainer = Trainer(args)
    trainer.train()
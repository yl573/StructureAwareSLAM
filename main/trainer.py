from main.data.data_loaders import PlaneNetDataLoader
from main.models.planenet import PlaneNet
from main.args import parse_args
import torch

from main.loss.losses import calc_plane_loss

class Trainer:

    def __init__(self, args):
        self.data_loader = PlaneNetDataLoader('/Volumes/MyPassport/planes_scannet_val.tfrecords', 'val', args.batchSize)
        self.model = PlaneNet(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.LR)
        print(args)

    def train(self):
        for i in range(10):
            batch = next(self.data_loader)

            planes, segmentation, depth = self.model(batch['image_norm'])

            plane_loss = calc_plane_loss(planes, batch['plane'])

            loss = torch.sum(plane_loss)

            print(loss.value())

            loss.backward()
            self.optimizer.step()


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()
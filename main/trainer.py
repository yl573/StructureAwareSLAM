from main.data.data_loaders import PlaneNetDataLoader
from main.models.planenet import PlaneNet
from main.args import parse_args
import torch
from tensorboardX import SummaryWriter
import os
import datetime
from main.loss.losses import calc_plane_loss


class Trainer:

    def __init__(self, args):
        self.args = args
        self.data_loader = PlaneNetDataLoader(args.val_path, 'val', args.batchSize)
        self.device = self.get_device()
        self.model = PlaneNet(args)
        if args.checkpoint:
            print('Loading state dict from: {}'.format(args.checkpoint))
            self.model.load_state_dict(torch.load(args.checkpoint))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.LR)
        self.save_dir = self.create_save_dir(args.log_dir, args.tag)
        print('Saving model data in: {}'.format(self.save_dir))
        self.tensorboard = SummaryWriter(log_dir=self.save_dir, comment=args.tag)

    def create_save_dir(self, log_dir, tag):
        timestamp = str(datetime.datetime.utcnow()).replace(' ', '_')
        dir_name = '{}_{}'.format(tag, timestamp)
        dir_path = os.path.join(log_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    def get_device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def batch_to_device(self, batch):
        for k, v, in batch.items():
            if type(v) == torch.Tensor:
                batch[k] = v.to(device=self.device)
        return batch

    def train(self):
        print('Training on device: {}'.format(self.device))

        for epoch in range(self.args.numEpochs):
            for i in range(int(self.args.numTrainingImages/self.args.batchSize)):
                current_iter = epoch * self.args.numTrainingImages + i

                batch = next(self.data_loader)
                batch = self.batch_to_device(batch)
                planes, segmentation, depth = self.model(batch['image_norm'])

                plane_loss = calc_plane_loss(planes, batch['plane'])

                loss = torch.sum(plane_loss)

                self.tensorboard.add_scalar('train/plane_loss', plane_loss.item(), current_iter)

                if i % self.args.printInterval == 0:
                    print('epoch: {}, iter: {}, loss: {}'.format(epoch, i, loss.item()))

                loss.backward()
                self.optimizer.step()

            save_path = os.path.join(self.save_dir, 'checkpoint-latest')
            torch.save(self.model.state_dict(), save_path)
            self.data_loader = iter(self.data_loader)



if __name__ == '__main__':
    args = parse_args()
    args.val_path = '/Volumes/MyPassport/planes_scannet_val.tfrecords'
    args.log_dir = '/Users/yuxuanliu/Desktop/4YP/StructureSLAM/logs'
    args.tag = 'test'
    args.save_dir = '/Users/yuxuanliu/Desktop/4YP/StructureSLAM/logs/models'
    # args.checkpoint = '/Users/yuxuanliu/Desktop/4YP/StructureSLAM/logs/test_2018-11-17_01:04:44.301218/checkpoint-latest'
    args.checkpoint = None
    args.numTrainingImages = 10
    args.numEpochs = 10
    args.printInterval = 20
    trainer = Trainer(args)
    trainer.train()
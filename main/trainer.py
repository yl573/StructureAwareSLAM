from main.data.data_loaders import PlaneNetDataLoader
from main.models.planenet import PlaneNet
from main.args import parse_args
import torch
from tensorboardX import SummaryWriter
import os
import datetime
from main.loss.losses import calc_plane_loss, calc_seg_loss, find_plane_assignment, permute_planes, permute_segmentation
from main.loss.tracker import CompositeLossTracker
import time


class Timer:

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        end = time.clock()
        interval = end - self.start
        # print('{} took {} seconds'.format(self.name, interval))


class Trainer:

    def __init__(self, args):
        self.args = args
        self.data_loader = PlaneNetDataLoader(args.val_path, 'val', args.batchSize)
        assert args.numTrainingImages <= len(self.data_loader)

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
        self.losses = CompositeLossTracker(['total_train_loss', 'plane_loss', 'seg_loss'])

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

    def print_losses(self):
        for k, v in self.losses.mean().items():
            print('{}: {}'.format(k, v))
        self.losses.reset()
        print()

    def train(self):
        print('Training on device: {}'.format(self.device))

        for epoch in range(self.args.numEpochs):
            for i in range(int(self.args.numTrainingImages / self.args.batchSize)):
                current_iter = epoch * self.args.numTrainingImages + i

                batch = next(self.data_loader)
                batch = self.batch_to_device(batch)

                with Timer('forward pass') as t:
                    planes, segmentation, depth = self.model(batch['image_norm'])

                assignment = find_plane_assignment(planes, batch['plane'])

                ordered_planes = permute_planes(planes, assignment)
                plane_loss = calc_plane_loss(ordered_planes, batch['plane'], batch['num_planes'])

                ordered_seg = permute_segmentation(segmentation, assignment)
                seg_loss = calc_seg_loss(ordered_seg, batch['segmentation_raw'])

                self.tensorboard.add_scalar('train/plane_loss', plane_loss.item(), current_iter)
                self.tensorboard.add_scalar('train/segmentation_loss', seg_loss.item(), current_iter)

                # loss = plane_loss + seg_loss
                loss = seg_loss

                if i % self.args.printInterval == 0:
                    print('epoch: {}, iter: {}, loss: {:.3f}'.format(epoch, i, loss.item()))

                loss.backward()
                self.optimizer.step()

                self.losses.update(dict(
                    total_train_loss=loss.item(),
                    plane_loss=plane_loss.item(),
                    seg_loss=seg_loss.item()
                ))

            print('\nepoch {} finished'.format(epoch))
            self.print_losses()
            self.tensorboard.add_scalar('train/total_loss', loss.item(), epoch)

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
    args.numTrainingImages = 100
    args.numEpochs = 5
    args.printInterval = 20

    args.drn_channels = (4, 8, 16, 32, 64, 64, 64, 64)
    args.drn_out_map = 32
    args.pyr_mid_planes = 32
    args.feat_planes = 64

    trainer = Trainer(args)
    trainer.train()

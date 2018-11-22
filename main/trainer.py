from main.data.data_loaders import PlaneNetDataLoader
from main.models.planenet import PlaneNet
from main.args import parse_args
from tensorboardX import SummaryWriter
import os
import datetime
from main.loss.losses import *
from main.loss.tracker import CompositeLossTracker
from main.visualization.plane_vis import draw_vis
from main.utils import Timer


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
        self.losses = CompositeLossTracker(['total_train_loss', 'plane_loss', 'seg_loss', 'depth_loss'])

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

    def seg_to_onehot(self, seg, dims):
        shape = [seg.size(0), dims, seg.size(1), seg.size(2)]
        batch_seg_scatter = torch.zeros(shape).scatter_(1, seg.unsqueeze(1).long().cpu(), 1)
        return batch_seg_scatter

    def train(self):
        print('Training on device: {}'.format(self.device))

        current_iter = 0
        for epoch in range(self.args.numEpochs):
            for i in range(int(self.args.numTrainingImages / self.args.batchSize)):
                current_iter += 1

                with Timer('load data') as t:
                    batch = next(self.data_loader)
                batch = self.batch_to_device(batch)

                with Timer('forward pass') as t:
                    planes_pred, seg_pred, depth_pred = self.model(batch.image_norm)

                # assignment = find_plane_assignment(planes_pred, batch.planes)

                assignment = find_plane_assignment_from_seg(seg_pred, batch.seg)

                ordered_planes = permute_planes(planes_pred, assignment)
                plane_loss = calc_plane_loss(ordered_planes, batch.planes, batch.num_planes)

                ordered_seg = permute_segmentation(seg_pred, assignment)
                seg_loss = calc_seg_loss(ordered_seg, batch.seg)

                all_depth_pred = calc_all_depth(depth_pred, ordered_planes, ordered_seg, batch.calib,
                                                batch.cam_height, batch.cam_width)

                batch_seg_onehot = self.seg_to_onehot(batch.seg, seg_pred.size(1))

                all_depth_gt = calc_all_depth(batch.depth, batch.planes, batch_seg_onehot, batch.calib,
                                              batch.cam_height,
                                              batch.cam_width)

                depth_loss = calc_depth_loss(all_depth_pred, batch.depth)

                loss = (self.args.plane_weight * plane_loss +
                        self.args.seg_weight * seg_loss +
                        self.args.depth_weight * depth_loss)

                with Timer('backward pass and update') as t:
                    loss.backward()
                    self.optimizer.step()

                self.losses.update(dict(
                    total_train_loss=loss.item(),
                    plane_loss=plane_loss.item(),
                    seg_loss=seg_loss.item(),
                    depth_loss=depth_loss.item()
                ))

                if i % self.args.printInterval == 0:
                    print('epoch: {}, iter: {}, loss: {:.3f}'.format(epoch, i, loss.item()))

                    self.tensorboard.add_scalar('train/plane_loss', plane_loss.item(), current_iter)
                    self.tensorboard.add_scalar('train/segmentation_loss', seg_loss.item(), current_iter)
                    self.tensorboard.add_scalar('train/depth_loss', depth_loss.item(), current_iter)
                    self.tensorboard.add_scalar('train/total_loss', loss.item(), current_iter)

                    vis = draw_vis(batch.image_raw, ordered_seg, batch.seg, all_depth_pred, all_depth_gt)
                    self.tensorboard.add_image('train/plane_visualization', vis, current_iter)

                    if self.args.train_callback:
                        self.args.train_callback({
                            'plane_loss': plane_loss.item(),
                            'seg_loss': seg_loss.item(),
                            'depth_loss': depth_loss.item(),
                            'visualization': vis
                        })

            print('\nepoch {} finished'.format(epoch))
            self.print_losses()

            save_path = os.path.join(self.save_dir, 'checkpoint-latest')
            torch.save(self.model.state_dict(), save_path)
            self.data_loader = iter(self.data_loader)


if __name__ == '__main__':
    args = parse_args()
    # args.val_path = '/Volumes/MyPassport/planes_scannet_val.tfrecords'
    args.val_path = '/Users/yuxuanliu/Desktop/4YP/planes_scannet_val.tfrecords'
    args.log_dir = '/Users/yuxuanliu/Desktop/4YP/StructureSLAM/logs'
    args.tag = 'test'
    args.save_dir = '/Users/yuxuanliu/Desktop/4YP/StructureSLAM/logs/models'
    # args.checkpoint = '/Users/yuxuanliu/Desktop/4YP/StructureSLAM/logs/test_2018-11-17_01:04:44.301218/checkpoint-latest'
    args.checkpoint = None
    args.numTrainingImages = 700
    args.numEpochs = 5
    args.printInterval = 1
    args.batchSize = 2
    args.plane_weight = 1
    args.seg_weight = 1
    args.depth_weight = 1
    args.train_callback = None

    # args.drn_channels = (4, 8, 16, 32, 64, 64, 64, 64)
    # args.drn_channels = (4, 4, 4, 4, 4, 4, 4, 4)
    args.drn_channels = channels = (16, 32, 64, 128, 256, 512, 512, 512)
    args.drn_out_map = 32
    args.pyr_mid_planes = 32
    args.feat_planes = 64

    trainer = Trainer(args)
    trainer.train()

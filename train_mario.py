
from main.args import parse_args
from main.trainer import Trainer


if __name__ == '__main__':
    args = parse_args()
    args.train_path = '../planes_scannet_val.tfrecords'
    # args.val_path = '/Users/yuxuanliu/Desktop/4YP/planes_scannet_val.tfrecords'
    args.log_dir = '../logs'
    args.tag = 'mario-{}'.format(args.name)
    args.checkpoint = None
    args.ordering = 'plane'
    args.gt_seg = False
    args.gt_planes = False
    args.numTrainingImages = 750
    args.numEpochs = 50
    args.printInterval = 16
    args.batchSize = 4
    args.plane_weight = 1
    args.seg_weight = 0.1
    args.depth_weight = 0.01
    args.adaptive_weights = True
    args.train_callback = None
    args.LR = 0.0003

    trainer = Trainer(args)
    trainer.train()
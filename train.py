from datasets import *
from utils.functions import MovingAverage, SavePath, ProgressBar
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss
from STMask import STMask
import os
import time
import torch
import torch.optim as optim
import torch.utils.data as data
import argparse
import datetime

import torch.distributed as dist

# Oof
import eval as eval_script
import eval_coco as eval_coco_script


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training Script')
    parser.add_argument('--config', default='STMask_plus_base_config',
                        help='The config object to use.')
    parser.add_argument('--port', default=None, type=str,
                        help='port for dist')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local_rank for distributed training')
    parser.add_argument('--is_distributed', dest='is_distributed', action='store_true',
                        help='use distributed for training')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for training')
    parser.add_argument('--eval_batch_size', default=1, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from. If this is "interrupt"' \
                         ', the model will resume training from the interrupt file.')
    parser.add_argument('--start_iter', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be' \
                         'determined from the file name.')
    parser.add_argument('--start_epoch', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be' \
                             'determined from the file name.')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Number of workers used in data_loading')
    parser.add_argument('--backbone.pred_scales_num', default=3, type=int,
                        help='Number of pred scales in backbone for getting anchors')
    parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                        help='Initial learning rate. Leave as None to read this from the config.')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum for SGD. Leave as None to read this from the config.')
    parser.add_argument('--decay', '--weight_decay', default=0.0001, type=float,
                        help='Weight decay for SGD. Leave as None to read this from the config.')
    parser.add_argument('--gamma', default=None, type=float,
                        help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')
    parser.add_argument('--save_folder', default='weights/weights_temp/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_interval', default=3, type=int,
                        help='The number of epochs between saving the model.')
    parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                        help='Only keep the latest checkpoint instead of each one.')
    parser.add_argument('--keep_latest_interval', default=100000, type=int,
                        help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help='Don\'t log per iteration information into log_folder.')
    parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                        help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
    parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                        help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')
    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True)

    args = parser.parse_args()

    return args


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))


loss_types = ['B', 'BIoU', 'center', 'Rep', 'C', 'C_focal', 'stuff', 'M_bce', 'M_dice', 'M_coeff',
              'T', 'B_shift', 'BIoU_shift', 'M_shift', 'P', 'D', 'S', 'I']


def train():
    if args.is_distributed:
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        torch.cuda.set_device(torch.cuda.current_device())
        device = torch.cuda.current_device()

    if not os.path.exists(args.save_folder) and args.local_rank == 0:
        os.mkdir(args.save_folder)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    net = STMask()
    net.train()

    if args.log:
        log = Log(cfg.name, args.save_folder, dict(args._get_kwargs()),
                  overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_folder)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    if args.resume is not None:
        if args.is_distributed:
            if args.local_rank == 0:
                print('Resuming training on local _rank=0, loading {}...'.format(args.resume))
                net.load_weights(path=args.resume)
        else:
            print('Resuming training, loading {}...'.format(args.resume))
            net.load_weights(path=args.resume)
        cur_lr = args.lr

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
            args.start_epoch = SavePath.from_str(args.resume).epoch
    else:
        if cfg.data_type == 'coco':
            print('Initializing weights based ImageNet ...')
            net.init_weights(backbone_path='weights/pretrained_models_coco/' + cfg.backbone.path,
                             local_rank=args.local_rank)
        else:
            print('Initializing weights based COCO ...')
            net.init_weights_coco(backbone_path='weights/pretrained_models_coco/' + cfg.backbone.path,
                                  local_rank=args.local_rank)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    criterion = MultiBoxLoss(net=net,
                             num_classes=cfg.num_classes,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold)

    train_dataset = get_dataset(cfg.data_type, cfg.train_dataset, cfg.backbone.transform)

    if args.is_distributed:
        if args.batch_size < 8:
            if args.local_rank == 0:
                print('Batch size of each gpu is:', args.batch_size, 'turn on sync batchnorm for DDP')
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net.to(device), device_ids=[args.local_rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        netloss = CustomDataParallel(NetLoss(net, criterion))
        netloss = netloss.cuda()
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        train_sampler = None

    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=cfg.detection_collate,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  pin_memory=True)

    # loss counters
    iteration = max(args.start_iter, 0)
    start_epoch = max(args.start_epoch, 0)
    last_time = time.time()
    if args.is_distributed:
        epoch_size = len(train_dataset) // (torch.cuda.device_count() * args.batch_size)
    else:
        epoch_size = len(train_dataset) // args.batch_size
    max_iter = epoch_size * cfg.max_epoch

    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types  # Forms the print order
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    if args.local_rank == 0:
        print('Begin training!')
        print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(start_epoch, cfg.max_epoch):

            # for datum in data_loader:
            for i, data_batch in enumerate(data_loader):
                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    cur_lr = (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init
                    set_lr(optimizer, cur_lr)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and epoch == cfg.lr_steps[step_index] and i == 0:
                    step_index += 1
                    cur_lr = args.lr * (args.gamma ** step_index)
                    set_lr(optimizer, cur_lr)

                optimizer.zero_grad()
                if args.is_distributed:
                    imgs, img_meta, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds = \
                        prepare_data_dist(data_batch, device)
                    preds = net(imgs)
                    losses = criterion(imgs, preds, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds)

                else:
                    losses = netloss(data_batch)
                    losses = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel

                loss = sum([losses[k] for k in losses])  # same weights in three sub-losses
                loss.backward()  # Do this to free up vram even if loss is not finite
                optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time = time.time()
                elapsed = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                if iteration % 10 == 0:
                    eta_str = \
                         str(datetime.timedelta(seconds=(max_iter - iteration) * time_avg.get_avg())).split('.')[0]

                    total = sum([loss_avgs[k].get_avg() for k in losses])
                    loss_labels = sum([[k, loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])

                    print(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' Total: %.3f || ETA: %s || timer: %.3f')
                          % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed]), flush=True)

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['Total'] = round(loss.item(), precision)
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                            lr=round(cur_lr, 10), elapsed=elapsed)

                if iteration % args.save_interval == 0 and iteration > 0 and args.local_rank == 0:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, epoch:', epoch)
                    torch.save(net.state_dict(), save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)

                    do_inference = True
                    if do_inference:
                        if cfg.data_type == 'coco':
                            setup_eval_coco()
                            compute_validation_map_coco(epoch, iteration, net, cfg.valid_dataset, log if args.log else None)
                        else:
                            print('Inference for valid data!')
                            setup_eval()
                            # valid_sub, the last one ten of training data
                            if cfg.data_type == 'vis':
                                cfg.valid_sub_dataset.has_gt = False
                                valid_sub_ann_file = cfg.valid_sub_dataset.ann_file
                                valid_sub_dataset = get_dataset(cfg.data_type, cfg.valid_sub_dataset, cfg.backbone.transform)
                            else:
                                cfg.valid_dataset.has_gt = False
                                valid_sub_ann_file = cfg.valid_dataset.ann_file
                                valid_sub_dataset = get_dataset(cfg.data_type, cfg.valid_dataset, cfg.backbone.transform)

                            compute_validation_map(net, valid_sub_dataset, ann_file=valid_sub_ann_file, epoch=epoch)

                            # valid or test datasets
                            if cfg.data_type == 'vis':
                                valid_dataset = get_dataset(cfg.data_type, cfg.valid_dataset, cfg.backbone.transform)
                                compute_validation_map(net, valid_dataset, epoch=epoch)

                iteration += 1

    except KeyboardInterrupt:
        print('Stopping early. Saving network...')

        # Delete previous copy of the interrupted network so we don't spam the weights folder
        SavePath.remove_interrupt(args.save_folder)

        torch.save(net.state_dict(), save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    torch.save(net.state_dict(), save_path(epoch, iteration))


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def prepare_data_dist(data_batch, device):
    with torch.no_grad():
        imgs, targets, masks, num_crowds = data_batch
        imgs = imgs.cuda(device)
        if cfg.data_type == 'coco':
            # padding images with different scales
            masks_list = [mask.cuda(device) for mask in masks]
            bboxes_list = [target[:, :4].cuda(device) for target in targets]
            labels_list = [target[:, -1].cuda(device) for target in targets]
            num_crowds = data_batch[3]
            ids_list, images_meta_list = None, None

        elif cfg.data_type == 'vis':
            images_meta_list = data_batch[1]
            bboxes_list = [box.cuda(device) for box in data_batch[2][0]]
            labels_list = [label.cuda(device) for label in data_batch[2][1]]
            masks_list = [mask.cuda(device) for mask in data_batch[2][2]]
            ids_list = [id.cuda(device) for id in data_batch[2][3]]
            num_crowds = None

        return imgs, images_meta_list, bboxes_list, labels_list, masks_list, ids_list, num_crowds


def compute_validation_loss(net, data_loader, dataset_size):
    global loss_types
    print()
    print('compute_validation_loss, needs few minutes ...')

    with torch.no_grad():
        losses = {}
        progress_bar = ProgressBar(30, dataset_size)

        # Don't switch to eval mode because we want to get losses
        iterations, results = 0, []
        for i, data_batch in enumerate(data_loader):
            _losses = net(data_batch)

            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            progress = (i + 1) / dataset_size * 100
            progress_bar.set_val(i + 1)
            print('\rProcessing Images  %s %6d / %6d (%5.2f%%)'
                  % (repr(progress_bar), i + 1, dataset_size, progress), end='')

            iterations += 1

        for k in losses:
            losses[k] /= iterations

        loss_labels = sum([[k, losses[k]] for k in loss_types if k in losses], [])
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

    return loss_labels


def compute_validation_map(net, dataset, ann_file=None, epoch=-1):
    with torch.no_grad():
        net.eval()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        if cfg.clip_prediction_module:
            eval_script.evaluate_clip(net, dataset, ann_file=ann_file, epoch=epoch)
        else:
            eval_script.evaluate(net, dataset, ann_file=ann_file, epoch=epoch)

        net.train()


def compute_validation_map_coco(epoch, iteration, net, dataset_config, log: Log = None):
    with torch.no_grad():
        net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_coco_script.evaluate(net, dataset_config, train_mode=True, epoch=epoch, iteration=iteration)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        net.train()


def setup_eval():
    eval_type = 'bbox' if cfg.data_type == 'vid' else 'segm'
    eval_script.parse_args(['--batch_size=' + str(args.eval_batch_size),
                            '--score_threshold=' + str(cfg.eval_conf_thresh),
                            '--save_folder=' + args.save_folder,
                            '--eval_types=' + eval_type
                            ])


def setup_eval_coco():
    eval_coco_script.parse_args(['--save_path='+str(args.save_folder)+'/valid_metrics.txt'])


if __name__ == '__main__':
    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    args = parse_args()
    cfg.data_type = 'vis'
    cfg.detection_collate = detection_collate_vis
    if args.config is not None:
        set_cfg(args.config)
        if 'coco' in args.config:
            cfg.data_type = 'coco'
            cfg.detection_collate = detection_collate_coco
        elif 'VID' in args.config:
            cfg.data_type = 'vid'
            cfg.detection_collate = detection_collate_vid

    # This is managed by set_lr
    cur_lr = args.lr
    replace('lr')
    replace('decay')
    replace('gamma')
    replace('momentum')
    replace('backbone.pred_scales_num')

    # cfg = Config.fromfile(args.config)
    train()

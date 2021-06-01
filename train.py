from datasets import *
from utils.functions import MovingAverage, SavePath, ProgressBar
from utils.logger import Log
from utils.augmentations import SSDAugmentation, BaseTransform
from utils import timer
from layers.modules import MultiBoxLoss
from STMask import STMask
import os
import time
import torch
from datasets import get_dataset, prepare_data
import torch.optim as optim
import torch.utils.data as data
import argparse
import datetime

# dist
import torch.multiprocessing as mp
import torch.distributed as dist
# import torch.nn.parallel.DistributedDataParallel as DDP

# Oof
import eval as eval_script
import eval_coco as eval_coco_script


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
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
parser.add_argument('--start_iter', default=0, type=int,
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
parser.add_argument('--config', default='STMask_plus_base_config',
                    help='The config object to use.')
parser.add_argument('--save_interval', default=3, type=int,
                    help='The number of epochs between saving the model.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1, do no validation.')
parser.add_argument('--output_json', dest='output_json', action='store_true',
                    help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
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

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

train_type = 'vis'
if args.config is not None:
    set_cfg(args.config)
    if 'coco' in args.config:
        train_type = 'coco'


# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')
replace('backbone.pred_scales_num')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

loss_types = ['B', 'BIoU', 'C', 'stuff', 'M', 'MIoU', 'T', 'center', 'B_shift', 'M_shift', 'M_occluded',
              'P', 'D', 'S', 'I']


def train():
    if args.is_distributed:
        os.environ['MASTER_PORT'] = args.port
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)

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
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(path=args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
    else:
        if train_type == 'coco':
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
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)
    if train_type == 'coco':
        train_dataset = COCODetection(image_path=cfg.dataset.train_images,
                                      info_file=cfg.dataset.train_info,
                                      transform=SSDAugmentation(MEANS))
        detection_collate = detection_collate_coco
    else:
        train_dataset = get_dataset(cfg.train_dataset)
        detection_collate = detection_collate_vis

    if args.is_distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        train_sampler = None

    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=detection_collate,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  pin_memory=True)

    # loss counters
    iteration = max(args.start_iter, 0)
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
        for epoch in range(cfg.max_epoch):

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

                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()
                if train_type == 'vis':
                    num_crowds = None
                    if args.is_distributed:
                        imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, img_meta = prepare_data(data_batch,
                                                                                              devices=args.local_rank)
                    else:
                        imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, img_meta = prepare_data(data_batch,
                                                                                              devices=torch.cuda.current_device())
                else:
                    gt_ids = None
                    if args.is_distributed:
                        imgs, gt_bboxes, gt_labels, gt_masks, num_crowds = prepare_data(data_batch,
                                                                                        train_type=train_type,
                                                                                        devices=args.local_rank)
                    else:
                        imgs, gt_bboxes, gt_labels, gt_masks, num_crowds = prepare_data(data_batch,
                                                                                        train_type=train_type,
                                                                                        devices=torch.cuda.current_device())

                preds = net(imgs)
                losses = criterion(preds, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds)
                if not args.is_distributed:
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

                # if i == 0 and epoch > 0:
                #     if epoch % args.save_interval == 0 and args.local_rank == 0:
                if iteration == 200 and args.local_rank == 0:
                        if args.keep_latest:
                            latest = SavePath.get_latest(args.save_folder, cfg.name)

                        print('Saving state, epoch:', epoch)
                        torch.save(net.state_dict(), save_path(epoch-1, iteration))

                        if args.keep_latest and latest is not None:
                            if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                                print('Deleting old save...')
                                os.remove(latest)

                        # This is done per epoch
                        if args.validation_epoch > 0:
                            save_path_valid_metrics = save_path(epoch-1, iteration).replace('.pth', '.txt')

                            if train_type == 'vis':
                                setup_eval()
                                # valid_sub
                                cfg.valid_sub_dataset.test_mode = False
                                metrics = compute_validation_map(net, valid_data=False, device=torch.cuda.current_device(),
                                                                 output_metrics_file=save_path_valid_metrics)

                                # valid datasets
                                metrics_valid = compute_validation_map(net, valid_data=True,
                                                                       device=torch.cuda.current_device(),
                                                                       output_metrics_file=save_path_valid_metrics)
                            else:
                                setup_eval_coco()
                                val_dataset = COCODetection(image_path=cfg.dataset.valid_images,
                                                            info_file=cfg.dataset.valid_info,
                                                            transform=BaseTransform(MEANS))
                                compute_validation_map_coco(epoch, iteration, net, val_dataset, log if args.log else None)


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


def compute_validation_map(net, valid_data=False, device=0, output_metrics_file=None):
    with torch.no_grad():
        net.eval()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        metrics = eval_script.validation(net, device=device,
                                         valid_data=valid_data, output_metrics_file=output_metrics_file)

        net.train()

    return metrics


def compute_validation_map_coco(epoch, iteration, net, dataset, log: Log = None):
    with torch.no_grad():
        net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_coco_script.evaluate(net, dataset, train_mode=True, iteration=iteration)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        net.train()


def setup_eval():
    eval_script.parse_args(['--no_bar',
                            '--batch_size=' + str(args.eval_batch_size),
                            '--output_json',
                            '--score_threshold=' + str(cfg.eval_conf_thresh),
                            '--mask_det_file='+args.save_folder+'eval_mask_det.json'])


def setup_eval_coco():
    eval_coco_script.parse_args([
                                 # '--max_images='+str(args.eval_batch_size),
                                 '--save_path='+str(args.save_folder)+'/valid_metrics.txt'])


if __name__ == '__main__':
    train()

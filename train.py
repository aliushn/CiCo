from datasets import *
from utils.functions import MovingAverage, SavePath, ProgressBar
from utils.logger import Log
from utils import timer
from layers import MultiBoxLoss
from CoreNet import CoreNet
import os
import time
import torch
import torch.optim as optim
import torch.utils.data as data
import argparse
import datetime
import torch.distributed as dist
from configs.load_config import load_config
import eval as eval_script
import eval_coco as eval_coco_script


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training Script')
    parser.add_argument('--config', default='configs/CiCo/cico_yolact_r50_yt19.py',
                        help='The config object to use.')
    parser.add_argument('--LR', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('--IMS_PER_BATCH', default=6, type=int,
                        help='number of frames in a clip')
    parser.add_argument('--NUM_CLIP_FRAMES', default=3, type=int,
                        help='number of frames in a clip')
    parser.add_argument('--OVERLAP_FRAMES', default=1, type=int,
                        help='number of overlap frames during inference.')
    parser.add_argument('--backbone_dir', default='outputs/pretrained_models/',
                        help='The directory of storing the pretrained models.')
    parser.add_argument('--is_distributed', dest='is_distributed', action='store_true',
                        help='use distributed for training')
    parser.add_argument('--port', default=None, type=str,
                        help='port for dist')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local_rank for distributed training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from. If this is "interrupt", '
                             'the model will resume training from the interrupt file.')
    parser.add_argument('--do_inference', dest='do_inference', action='store_false',
                        help='If True, you need to split the train data into two parts: train.json and valid_sub.json. '
                             'the former for training and the latter for inference during training.')
    parser.add_argument('--start_iter', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be determined '
                             'from the file name.')
    parser.add_argument('--start_epoch', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be determined '
                             'from the file name.')
    parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                        help='Only keep the latest checkpoint instead of each one.')
    parser.add_argument('--keep_latest_interval', default=10000, type=int,
                        help='When --keep_latest is on, don\'t delete the latest file at these intervals. '
                             'This should be a multiple of save_interval or 0.')
    parser.add_argument('--no_log', dest='log', action='store_false',
                        help='Don\'t log per iteration information into log_folder.')
    parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                        help='CiCo will automatically scale the lr and the number of iterations depending '
                             'on the batch size. Set this if you want to disable that.')
    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True)

    args = parser.parse_args()

    return args


def train(cfg):
    if args.is_distributed:
        # TODO: distributed training may degrade performance, test them
        os.environ['MASTER_PORT'] = args.port
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        torch.cuda.set_device(torch.cuda.current_device())
        device = torch.cuda.current_device()

    cfg.freeze_bn = True if cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.NUM_CLIPS < 4 else False
    print('Disabling batch norm:', cfg.freeze_bn)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    net = CoreNet(cfg)
    net.train()

    # Update configs
    cfg.NAME = cfg.NAME+'_f'+str(cfg.SOLVER.NUM_CLIP_FRAMES)
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    print('Output directory is: {}'.format(cfg.OUTPUT_DIR))
    if not os.path.exists(cfg.OUTPUT_DIR) and args.local_rank == 0:
        os.mkdir(cfg.OUTPUT_DIR)
    if args.log:
        log = Log(cfg.NAME, cfg.OUTPUT_DIR, cfg, overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs.
    timer.disable_all()

    # Both of these can set args.resume to None, so do them before the check
    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(cfg.OUTPUT_DIR)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(cfg.OUTPUT_DIR, cfg.NAME)
    if args.resume is not None:
        if args.is_distributed:
            if args.local_rank == 0:
                print('Resuming training on local _rank=0, loading {}...'.format(args.resume))
                net.load_weights(path=args.resume)
        else:
            print('Resuming training, loading {}...'.format(args.resume))
            net.load_weights(path=args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration
            args.start_epoch = SavePath.from_str(args.resume).epoch

    else:
        bb_path = cfg.MODEL.BACKBONE.SWINT.path if cfg.MODEL.BACKBONE.SWINT.engine else cfg.MODEL.BACKBONE.PATH
        bb_path = args.backbone_dir + bb_path
        print('Initializing weights from', bb_path)
        if cfg.DATASETS.TYPE == 'coco':
            print('Initializing weights based ImageNet ...')
            net.init_weights(backbone_path=bb_path, local_rank=args.local_rank)
        else:
            print('Initializing weights based COCO ...')
            net.init_weights_coco(backbone_path=bb_path, local_rank=args.local_rank)

    # loss counters
    iteration = max(args.start_iter, 0)
    start_epoch = max(args.start_epoch, 0)
    adj_lr_times = sum([1 for step in list(cfg.SOLVER.LR_STEPS) if step <= start_epoch])
    cur_lr = cfg.SOLVER.LR * (cfg.SOLVER.GAMMA ** adj_lr_times)
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = adj_lr_times

    optimizer = optim.SGD(net.parameters(), lr=cur_lr, momentum=cfg.SOLVER.MOMENTUM,
                          weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    criterion = MultiBoxLoss(cfg, net=net,
                             num_classes=cfg.DATASETS.NUM_CLASSES,
                             pos_threshold=cfg.MODEL.PREDICTION_HEADS.POSITIVE_IoU_THRESHOLD,
                             neg_threshold=cfg.MODEL.PREDICTION_HEADS.NEGATIVE_IoU_THRESHOLD)
    train_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.TRAIN, cfg.INPUT,
                                cfg.SOLVER.NUM_CLIP_FRAMES, cfg.SOLVER.NUM_CLIPS)

    if args.is_distributed:
        if cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.NUM_CLIP_FRAMES < 8:
            if args.local_rank == 0:
                print('Batch size of each gpu is lower than 8, turn on sync batchnorm for DDP')
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = torch.nn.parallel.DistributedDataParallel(net.to(device), device_ids=[args.local_rank])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        # We use customed dataparallel to balance memory of multiple GPUs
        # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        netloss = CustomDataParallel(NetLoss(net, criterion))
        netloss = netloss.cuda()
        train_sampler = None

    # We support train models on coco, vis and vid datasets now, please indicate a specific data type
    if cfg.DATASETS.TYPE == 'coco':
        detection_collate = detection_collate_coco
    elif cfg.DATASETS.TYPE in {'vis'}:
        detection_collate = detection_collate_vis
    elif cfg.DATASETS.TYPE in {'vid', 'det'}:
        detection_collate = detection_collate_vid
    else:
        RuntimeError('Your dataset is not supported now!! Please choose from {coco, vis, vid, det}.')

    batch_size = cfg.SOLVER.IMS_PER_BATCH * torch.cuda.device_count()
    data_loader = data.DataLoader(train_dataset, batch_size,
                                  num_workers=cfg.SOLVER.NUM_WORKERS,
                                  collate_fn=detection_collate,
                                  sampler=train_sampler,
                                  shuffle=(train_sampler is None),
                                  pin_memory=True)

    last_time = time.time()
    epoch_size = len(train_dataset) // batch_size
    max_iter = epoch_size * cfg.SOLVER.MAX_EPOCH
    save_path = lambda epoch, iteration: SavePath(cfg.NAME, epoch, iteration).get_path(root=cfg.OUTPUT_DIR)
    time_avg = MovingAverage()

    global loss_types  # Forms the print order
    loss_types = ['B', 'BIoU', 'center', 'Rep', 'C', 'C_focal', 'M_bce', 'M_dice', 'M_coeff', 'T', 'T_kl',
                  'B_T2S', 'BIoU_T2S', 'M_T2S', 'P', 'D', 'S', 'I']
    loss_avgs = {k: MovingAverage(100) for k in loss_types}

    if args.local_rank == 0:
        print('Begin training!')
        print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):

            # for datum in data_loader:
            for i, data_batch in enumerate(data_loader):
                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.SOLVER.LR_WARMUP_UNTIL > 0 and iteration <= cfg.SOLVER.LR_WARMUP_UNTIL:
                    cur_lr = (cfg.SOLVER.LR-cfg.SOLVER.LR_WARMUP_INIT) * (iteration / cfg.SOLVER.LR_WARMUP_UNTIL) \
                             + cfg.SOLVER.LR_WARMUP_INIT
                    set_lr(optimizer, cur_lr)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.SOLVER.LR_STEPS) and epoch == cfg.SOLVER.LR_STEPS[step_index] and i == 0:
                    step_index += 1
                    cur_lr = cfg.SOLVER.LR * (cfg.SOLVER.GAMMA ** step_index)
                    set_lr(optimizer, cur_lr)

                optimizer.zero_grad()
                if args.is_distributed:
                    # TODO: double check whether it is work or not
                    imgs, img_meta, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds = \
                        prepare_data_dist(cfg.DATASETS.TYPE, data_batch, device)
                    preds = net(imgs)
                    losses = criterion(preds, gt_bboxes, gt_labels, gt_masks, gt_ids, num_crowds)

                else:
                    losses = netloss(data_batch)
                    losses = {k: v.mean() for k, v in losses.items()}  # Mean here because Dataparallel

                loss = sum([losses[k] for k in losses])  # same weights in three sub-losses
                loss.backward()  # Do this to free up vram even if loss is not finite
                optimizer.step()

                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                elapsed = time.time() - last_time
                last_time = time.time()

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

                if args.log and iteration % 500 == 0:
                    loss_info = {k: round(losses[k].item(), 5) for k in losses}
                    loss_info['Total'] = round(loss.item(), 5)
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration, lr=round(cur_lr, 10), elapsed=elapsed)

                # Save models and run valid set
                if iteration % int(0.5*cfg.SOLVER.SAVE_INTERVAL) == 0 and iteration > 0 and args.local_rank == 0:
                    if args.keep_latest:
                        latest = SavePath.get_latest(cfg.OUTPUT_DIR, cfg.DATASETS.TYPE)

                    print('Saving state, epoch:', epoch)
                    torch.save(net.state_dict(), save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != cfg.SOLVER.SAVE_INTERVAL:
                            print('Deleting old save...')
                            os.remove(latest)

                    # Because the ground-truth annotations of all valid and test data in VIS task are not published,
                    # we have to submit their results into Codelab for evaluation, please refer to the official web
                    # for more details, like YTVIS2019: https://youtube-vos.org/dataset/vis/.
                    # During training, we split train data into two parts: train and valid_sub, where the former
                    # (the first nine-tenths of train data) is used for training, while the latter
                    # (the last one-tenth of train data) is used for validation.
                    if args.do_inference:
                        if cfg.DATASETS.TYPE == 'coco':
                            setup_eval_coco()
                            compute_validation_map_coco(net, epoch, iteration, log if args.log else None)
                        else:
                            print('Inference for valid_sub data!')
                            setup_eval(epoch)
                            # valid_sub, the last one ten of training data for inference
                            valid_sub_dataset = get_dataset(cfg.DATASETS.TYPE, cfg.DATASETS.VALID_SUB, cfg.INPUT,
                                                            cfg.TEST.NUM_CLIP_FRAMES, num_clips=1, inference=True)
                            compute_validation_map(net, valid_sub_dataset)

                    # For valid or test datasets, we generate a results_epoch.json file on the output path,
                    # you need first move results_epoch.json to results.json and then form a .zip file,
                    # finally upload the .zip file to the Codelab for evaluation.
                    print('Inference for valid data!')
                    if cfg.DATASETS.TYPE == 'vis' and epoch > 6:
                        setup_eval(epoch)
                        valid_dataset = get_dataset(cfg.DATASETS.TYPE,  cfg.DATASETS.VALID, cfg.INPUT,
                                                    cfg.TEST.NUM_CLIP_FRAMES, num_clips=1, inference=True)
                        compute_validation_map(net, valid_dataset)

                iteration += 1

    except KeyboardInterrupt:
        print('Stopping early. Saving network into ', cfg.OUTPUT_DIR)

        # Delete previous copy of the interrupted network so we don't spam the weights folder
        SavePath.remove_interrupt(cfg.OUTPUT_DIR)
        torch.save(net.state_dict(), save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    torch.save(net.state_dict(), save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def prepare_data_dist(data_type, data_batch, device):
    with torch.no_grad():
        imgs, img_metas, anns = data_batch
        h, w = imgs.size()[-2:]
        imgs = imgs.reshape(-1, 3, h, w).cuda(device)
        from datasets.utils import GtList_from_tensor

        if data_type == 'coco':
            boxes, labels, masks, num_crowds = anns
            # padding images with different scales
            masks, boxes, img_metas = GtList_from_tensor(h, w, masks, boxes, img_metas)
            boxes_list = [torch.tensor(box).cuda(device) for box in boxes]
            labels_list = [torch.tensor(label).cuda(device) for label in labels]
            masks_list = [torch.tensor(mask).cuda(device) for mask in masks]
            ids_list = [None] * len(boxes)

        elif data_type == 'vis':
            boxes, labels, masks, obj_ids = anns
            # padding images with different scales
            masks, boxes, img_metas = GtList_from_tensor(h, w, masks, boxes, img_metas)
            boxes_list = [torch.from_numpy(box).float().cuda(device) for box in boxes]
            labels_list = [torch.from_numpy(label).cuda(device) for label in labels]
            masks_list = [torch.from_numpy(mask).cuda(device) for mask in masks]
            ids_list = [torch.from_numpy(id).cuda(device) for id in obj_ids]
            num_crowds = [0] * len(boxes)
        elif data_type == 'vid':
            boxes, labels, obj_ids, occluded_boxes = anns
            # padding images with different scales
            _, boxes, img_metas = GtList_from_tensor(h, w, None, boxes, img_metas)
            boxes_list = [torch.tensor(box).cuda(device) for box in boxes]
            labels_list = [torch.tensor(label).cuda(device) for label in labels]
            ids_list = [torch.tensor(id).cuda(device) for id in obj_ids]
            num_crowds = [torch.tensor(occluded).cuda(device) for occluded in occluded_boxes]
            masks_list = [None] * len(boxes)

        return imgs, img_metas, boxes_list, labels_list, masks_list, ids_list, num_crowds


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


def compute_validation_map(net, dataset):
    with torch.no_grad():
        net.eval()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        eval_script.evaldatasets(net, dataset, cfg.DATASETS.TYPE, cfg_input=cfg.INPUT, use_cico=cfg.CiCo.ENGINE)
        net.train()


def compute_validation_map_coco(net, epoch, iteration, log: Log = None):
    with torch.no_grad():
        net.eval()

        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_coco_script.evaluate(cfg, net, train_mode=True, epoch=epoch, iteration=iteration)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        net.train()


def setup_eval(epoch):
    eval_type = 'bbox' if cfg.DATASETS.TYPE == 'vid' else 'segm'
    if cfg.DATASETS.TYPE == 'vis' and not cfg.MODEL.TRACK_HEADS.TRAIN_TRACK:
        cfg.TEST.OVERLAP_FRAMES = max(cfg.TEST.OVERLAP_FRAMES, 1)
    eval_script.parse_args(['--batch_size=' + str(cfg.TEST.IMS_PER_BATCH),
                            '--eval_types=' + eval_type,
                            '--epoch=' + str(epoch),
                            '--overlap_frames=' + str(cfg.TEST.OVERLAP_FRAMES),
                            '--save_folder=' + str(cfg.OUTPUT_DIR),
                            '--NUM_CLIP_FRAMES='+str(cfg.SOLVER.NUM_CLIP_FRAMES)])


def setup_eval_coco():
    eval_coco_script.parse_args(['--batch_size='+str(cfg.TEST.IMS_PER_BATCH)])


if __name__ == '__main__':
    if torch.cuda.device_count() == 0:
        print('No GPUs detected. Exiting...')
        exit(-1)

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Update training parameters from the args if necessary
    def replace(name, dict_cfg):
        if getattr(args, name) is not None: setattr(dict_cfg, name, getattr(args, name))

    args = parse_args()
    cfg = load_config(args.config)
    cfg.NAME = args.config.split('/')[-1][:-3]
    replace('LR', cfg.SOLVER)
    replace('IMS_PER_BATCH', cfg.SOLVER)
    replace('NUM_CLIP_FRAMES', cfg.SOLVER)
    replace('NUM_CLIP_FRAMES', cfg.TEST)
    replace('OVERLAP_FRAMES', cfg.TEST)
    print('Configs of SOLVER:', cfg.SOLVER)
    print()
    print('Configs of TEST:', cfg.TEST)

    train(cfg)

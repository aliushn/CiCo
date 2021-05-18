from datasets import *
import os
import torch
from datasets import get_dataset, prepare_data
import torch.utils.data as data
import argparse

# Oof
import eval as eval_script
from collections import defaultdict
import matplotlib.pyplot as plt


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in data_loading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--config', default='STMask_plus_base_config',
                    help='The config object to use.')
parser.add_argument('--save_path', default='results/eval_mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')

parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()

if args.config is not None:
    set_cfg(args.config)

color_cache = defaultdict(lambda: {})


def train(mask_alpha=0.45):
    train_dataset = get_dataset(cfg.train_dataset)
    data_loader = data.DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)

    print()
    # try-except so you can use ctrl+c to save early and stop training
    # for datum in data_loader:
    for i, data_batch in enumerate(data_loader):
        imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, imgs_meta = prepare_data(data_batch, is_cuda=True,
                                                                               train_mode=True)
        imgs, gt_bboxes, gt_labels, gt_masks, gt_ids, imgs_meta = imgs[0], gt_bboxes[0], gt_labels[0], gt_masks[0], \
                                                                  gt_ids[0], imgs_meta[0]

        print(imgs.size(), gt_bboxes)
        batch_size = imgs[0].size(0)
        for bs in range(0, batch_size, 2):
            n_cur = gt_masks[bs].size(0)
            gt_masks_cur = gt_masks[bs][:, :, :, None].repeat(1, 1, 1, 3)
            # This is 1 everywhere except for 1-mask_alpha where the mask is
            inv_alph_masks = gt_masks_cur.sum(0) * (-mask_alpha) + 1
            gt_masks_cur_color = []

            # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
            for j in range(n_cur):
                colors = get_color(j, gt_ids, on_gpu=imgs.device.index).view(1, 1, 3)
                gt_masks_cur_color.append(gt_masks_cur[j] * colors * mask_alpha)

            gt_masks_cur_color = torch.stack(gt_masks_cur_color, dim=0).sum(0)
            img_color = imgs[bs].permute(1, 2, 0).contiguous() * inv_alph_masks + gt_masks_cur_color
            img_color = torch.stack(img_color, dim=0)
            img_numpy = img_color.cpu().numpy()
            video_id, frame_id = imgs_meta[bs]['video_id'], imgs_meta[bs]['frame_id']
            plt.imshow(img_numpy)
            plt.axis('off')
            plt.title(str(frame_id))

            root_dir = ''.join([args.save_path, '/out/', str(video_id), '/'])
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            plt.savefig(''.join([root_dir, str(frame_id), '.png']))
            plt.clf()


# Quick and dirty lambda for selecting the color for a particular index
# Also keeps track of a per-gpu color cache for maximum speed
def get_color(j, color_type, on_gpu=None, undo_transform=True):
    global color_cache
    color_idx = color_type[j] % len(cfg.COLORS)

    if on_gpu is not None and color_idx in color_cache[on_gpu]:
        return color_cache[on_gpu][color_idx]
    else:
        color = cfg.COLORS[color_idx]
        if not undo_transform:
            # The image might come in as RGB or BRG, depending
            color = (color[2], color[1], color[0])
        if on_gpu is not None:
            color = torch.Tensor(color).to(on_gpu).float() / 255.
            color_cache[on_gpu][color_idx] = color
        return color

if __name__ == '__main__':
    train()

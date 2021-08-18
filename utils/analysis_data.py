from datasets import *
import torch
import torch.utils.data as data
from layers.utils import center_size


def main(config, data_type='vis'):
	if config is not None:
		set_cfg(config)

	cfg.train_dataset.has_gt = True
	dataset = get_dataset(data_type, cfg.train_dataset, cfg.backbone.transform)
	data_loader = data.DataLoader(dataset, 4,
                                  shuffle=False,
                                  collate_fn=detection_collate_vis,
                                  pin_memory=True)
	#
	hist_area = torch.arange(0, 960, 24) ** 2
	count = torch.zeros(len(hist_area))
	results = []

	for it, data_batch in enumerate(data_loader):
		imgs_batch, img_metas, (boxes, labels, masks, ids) = data_batch
		h, w = imgs_batch.size()[-2:]
		bs = imgs_batch.size(0)
		for i in range(bs):
			box_c = center_size(boxes[i])
			box_area = box_c[:, 2] * h * box_c[:, 3] * w
			g = box_area.unsqueeze(0) > hist_area.unsqueeze(1)
			g[:-1] += g[1:]
			count += (g == 1).sum(dim=1)

	print('Done!')


if __name__ == '__main__':
	config = 'STMask_plus_resnet50_OVIS_config'
	main(config)

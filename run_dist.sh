CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python \-m torch.distributed.launch --nproc_per_node=2  train.py --config=configs/VIS/r50_base_YTVIS2019_cubic_3D_c5_1X --port=200957 --is_distributed

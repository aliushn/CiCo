CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_17_50000.pth --eval_data=valid_sub --batch_size=1 --overlap_frames=0 --epoch=0

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_17_50000.pth --eval_data=valid_sub --batch_size=1 --overlap_frames=1 --epoch=1

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_17_50000.pth --eval_data=valid --batch_size=1 --overlap_frames=0 --epoch=0

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_17_50000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1


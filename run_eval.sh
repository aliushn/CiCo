CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r101_f3_multiple_yt21.py --trained_model=outputs/YTVIS2021/r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X/r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X_11_50000.pth  --eval_data=valid_sub --batch_size=1 --overlap_frames=1 --epoch=11

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r101_f3_multiple_yt21.py --trained_model=outputs/YTVIS2021/r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X/r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X_11_50000.pth  --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=11

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_swint_f3_dynamic_mask_head_multiple_yt19.py  --trained_model=outputs/YTVIS2019/cico_swint_f3_dynamic_mask_head_multiple_yt19_pergpu3/cico_swint_f3_dynamic_mask_head_multiple_yt19_11_65000.pth --eval_data=valid_sub --batch_size=1 --overlap_frames=1 --epoch=1

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis_11_55000.pth --eval_data=valid_sub --batch_size=1 --overlap_frames=1 --epoch=1

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis_11_55000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_17_50000.pth --eval_data=valid --batch_size=1 --overlap_frames=0 --epoch=0

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_17_50000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1


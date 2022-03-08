# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r101_f3c2_multiple_ohem_ovis.py --trained_model=outputs/OVIS/r101_base_OVIS_cubic_3D_c3s2_indbox_multiple_ohem_1X/r101_base_OVIS_cubic_3D_c3s2_indbox_multiple_ohem_1X_10_50000.pth  --eval_data=valid_sub --batch_size=1 --overlap_frames=1 --epoch=10

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r101_f3c2_multiple_ohem_ovis.py --trained_model=outputs/OVIS/r101_base_OVIS_cubic_3D_c3s2_indbox_multiple_ohem_1X/r101_base_OVIS_cubic_3D_c3s2_indbox_multiple_ohem_1X_10_50000.pth  --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=10

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r101_f3_multiple_yt21.py --trained_model=outputs/YTVIS2021/r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X/r101_base_YTVIS2021_cubic_3D_c3_indbox_multiple_1X_11_50000.pth  --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=12

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3_dynamic_mask_head_multiple_yt21.py  --trained_model=outputs/YTVIS2021/cico_r50_f3_dynamic_mask_head_multiple_yt21/cico_r50_f3_dynamic_mask_head_multiple_yt21_10_40000.pth --eval_data=valid --batch_size=1 --overlap_frames=2 --epoch=12

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis/cico_r50_f3c2_dynamic_mask_head_multiple_ohem_ovis_11_55000.pth --eval_data=valid --batch_size=1 --overlap_frames=2 --epoch=12

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_swint_f3_multiple_ovis.py  --trained_model=outputs/OVIS/cico_swint_f3_multiple_ovis/cico_swint_f3_multiple_ovis_11_55000.pth --eval_data=valid_sub --batch_size=1 --overlap_frames=1 --epoch=11

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_swint_f3_multiple_ovis.py  --trained_model=outputs/OVIS/cico_swint_f3_multiple_ovis/cico_swint_f3_multiple_ovis_11_55000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=11

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r101_f3c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r101_f3c2_multiple_ohem_ovis/cico_r101_f3c2_multiple_ohem_ovis_11_55000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=11

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f6_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f6_multiple_ohem_ovis/cico_r50_f6_multiple_ohem_ovis_11_40000.pth --eval_data=valid --batch_size=1 --overlap_frames=2 --epoch=2

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f7_multiple_yt19.py  --trained_model=outputs/YTVIS2019/cico_r50_f7_multiple_yt19/cico_r50_f7_multiple_yt19_10_55000.pth --eval_data=valid --batch_size=1 --overlap_frames=0 --epoch=1 --display

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5_multiple_sparse_yt19.py  --trained_model=outputs/YTVIS2019/cico_r50_f5_multiple_sparse_yt19/cico_r50_f5_multiple_sparse_yt19_11_45000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3_int1_boxclass_cphmh_yt19.py  --trained_model=outputs/YTVIS2019/cico_r50_f3_int1_cir3_boxtrack_cphcmh_yt19/cico_r50_f3_int1_boxtrack_cphcmh_yt19_10_27500.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_11_60000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=11

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f4c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f4c2_multiple_ohem_ovis/cico_r50_f4c2_multiple_ohem_ovis_11_40000.pth --eval_data=valid --batch_size=1 --overlap_frames=2 --epoch=12

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_11_60000.pth --eval_data=valid --batch_size=1 --overlap_frames=3 --epoch=13

# CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f5c2_multiple_ohem_ovis.py  --trained_model=outputs/OVIS/cico_r50_f5c2_multiple_ohem_ovis/cico_r50_f5c2_multiple_ohem_ovis_11_60000.pth --eval_data=valid --batch_size=1 --overlap_frames=4 --epoch=14


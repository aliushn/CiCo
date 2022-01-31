CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3_multiple_yt21.py  --trained_model=outputs/YTVIS2021/cico_r50_f3_multiple_yt21/cico_r50_f3_multiple_yt21_11_45000.pth --eval_data=valid --batch_size=1 --overlap_frames=0 --epoch=0

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_r50_f3_multiple_yt21.py  --trained_model=outputs/YTVIS2021/cico_r50_f3_multiple_yt21/cico_r50_f3_multiple_yt21_11_45000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_swint_f3_multiple_yt21.py  --trained_model=outputs/YTVIS2021/cico_swint_f3_multiple_yt21/cico_swint_f3_multiple_yt21_10_80000.pth --eval_data=valid --batch_size=1 --overlap_frames=0 --epoch=0

CUDA_VISIBLE_DEVICES=0 python eval.py --config=configs/CiCo/cico_swint_f3_multiple_yt21.py  --trained_model=outputs/YTVIS2021/cico_swint_f3_multiple_yt21/cico_swint_f3_multiple_yt21_10_80000.pth --eval_data=valid --batch_size=1 --overlap_frames=1 --epoch=1



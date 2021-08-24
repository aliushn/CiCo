CUDA_VISIBLE_DEVICES=1 python eval.py --save_folder=weights/OVIS/weights_plus_r50_m32_yolact_clip_pred_w_corr11_ext_ind_box_focal_768/ --trained_model=weights/OVIS/weights_plus_r50_m32_yolact_clip_pred_w_corr11_ext_ind_box_focal_768/STMask_plus_resnet50_OVIS_29_60000.pth --eval_data=valid --overlap_frames=1 --display=False --epoch=1


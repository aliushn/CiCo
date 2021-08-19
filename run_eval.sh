CUDA_VISIBLE_DEVICES=1 python eval.py --save_folder=weights/OVIS/weights_plus_r50_m32_yolact_clip_prediction_wo_corr_a3_640_ext_indbox_focal/ --trained_model=weights/OVIS/weights_plus_r50_m32_yolact_clip_prediction_wo_corr_a3_640_ext_indbox_focal/STMask_plus_resnet50_OVIS_30_80000.pth --eval_data=valid --overlap_frames=0 --display=False --epoch=0

CUDA_VISIBLE_DEVICES=1 python eval.py --save_folder=weights/OVIS/weights_plus_r50_m32_yolact_clip_prediction_wo_corr_a3_640_ext_indbox_focal/ --trained_model=weights/OVIS/weights_plus_r50_m32_yolact_clip_prediction_wo_corr_a3_640_ext_indbox_focal/STMask_plus_resnet50_OVIS_30_80000.pth --eval_data=valid --overlap_frames=1 --display=False --epoch=1 


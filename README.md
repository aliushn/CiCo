# CiCo
An offical code repository for our paper submitted in ECCV2022
- Clip-in Clip-out: An Alignment-free One-stage Video Instance Segmentation Approach 

  ![](https://github.com/MinghanLi/CiCo/tree/main/imgs/fifo_cico.png)
 
# Installation
 - Clone this repository 
   ```Shell
   git clone https://github.com/MinghanLi/CiCo.git
   cd CiCo
   ```
 - Set up the environment using [Anaconda](https://www.anaconda.com/distribution/):
   ```Shell
   conda env create -f environment.yml
   conda activate cico-env
    ```
       
 - According to your Cuda and pytorch version to install mmcv or mmcv-full from [here](https://github.com/open-mmlab/mmcv). 
   ```Shell
   # An example pytorch 1.10 and cuda 11.3 with mmcv version 1.4.2
   pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
   ```
 - Compile a customized COCO API for YouTubeVIS dataset from [here](https://github.com/youtubevos/cocoapi)
   ```Shell
   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   git clone https://github.com/youtubevos/cocoapi
   cd cocoapi/PythonAPI
   # To compile and install locally 
   python setup.py build_ext --inplace
   # To install library to Python site-packages 
   python setup.py build_ext install
   ```

## Run 

### Prepare datasets and models
 - Datasets: If you'd like to train or test CiCo, please download the datasets from the official web: [YTVIS2019](https://youtube-vos.org/dataset/), [YTVIS2021](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/).
   You can update your data path in `configs/_base_/datasets/vis.py`
   ```Shell
   cd CiCo
   vim configs/_base_/datasets/vis.py
   ```

 - CoCo-pretrained models with ResNet or Swin transformer backbones. To train, please put them in a directory, such as `outputs/pretrained_models/`.

### Inference
   ```Shell
   # add --display to display instance masks
   python eval.py --trained_model=path/to/your/trained/models.py --NUM_CLIP_FRAMES=3 --overlap_frames=1 
   ```

### Training
   ```Shell
   # Train CiCo with a 3-frame clip on 4 GPUs
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config=configs/CiCo/cico_yolact_r50_yt19.py --NUM_CLIP_FRAMES=3 
   --IMS_PER_BATCH=6 --LR=0.001 --backbone_dir=outputs/pretrained_models/
   ```

## Main Results

### Quantitative Results on YTVIS2019 
| Backbone    |FCN       | mAP  | Trained models | Results|
|:-----------:|:--------:|:----:|-----------------------------------------------------------------------------------------------------------------|-----------------|
| R50         |Yolact    | 37.1 |[cico_yolact_r50_yt19_f3.pth](https://drive.google.com/file/d/1tCxL1FbzhoSH9Dv2nPx2fG2KBb7HUoAB/view?usp=sharing)   | [stdout.txt](https://drive.google.com/file/d/11V_u5leyq2qqv50f1_tA99bYxeAncr76/view?usp=sharing)
| R50         |CondInst  | 37.3 |[cico_CondInst_r50_yt19_f3.pth](https://drive.google.com/file/d/1-pPSs4TFsttlOvd1YVX2heCimG0Dcyda/view?usp=sharing) | [stdout.txt](https://drive.google.com/file/d/11MBBchibbokVpqbLKIapSVWGuTYg9pdT/view?usp=sharing)
| R101        |Yolact    | 39.6 |
| R101        |CondInst  | 39.6 |[cico_CondInst_r101_yt19_f3.pth](https://drive.google.com/file/d/1h-i9LzZ1ThdI_AXQDyWmRyPGS-fMqPW1/view?usp=sharing)| [stdout.txt](https://drive.google.com/file/d/1z_XFMA_bllIFw-rPLjntC-XbiMpgp0it/view?usp=sharing)
| Swin-tiny   |Yoalct    | 41.8 |[cico_CondInst_yolact_yt19_f3.pth]()   |[stdout.txt](https://drive.google.com/file/d/1IpSLVYbqYa-C2ZQ9vKQmNTMbmTGH1Cdk/view?usp=sharing)
| Swin-tiny   |CondInst  | 41.4 |[cico_CondInst_swint_yt19_f3.pth](https://drive.google.com/file/d/1Z4zy3L4g12TmA5wEJCFCVRcacZCDm3nA/view?usp=sharing) | [stdout.txt](https://drive.google.com/file/d/1Rx6JiYUduWjgkxRvRzA5akXdfK056BFq/view?usp=sharing)

### Quantitative Results on YTVIS2021
| Backbone    |FCN       | mAP  | Weights | Results|
|:-----------:|:--------:|:----:|-----------------------------------------------------------------------------------------------------------------|-----------------|
| R50         |Yolact    | 35.2 |[cico_yolact_r50_yt21_f3.pth](https://drive.google.com/file/d/1qSxR_otaZ7UczTNEouyb-fFqPZeTTZk2/view?usp=sharing)    |[stdout.txt](https://drive.google.com/file/d/1MFDeYcZHBT5U8aa_jbb4saPOD1BZfqWA/view?usp=sharing)
| R50         |CondInst  | 35.4 |[cico_CondInst_r50_yt21_f3.pth](https://drive.google.com/file/d/1gT_KOXocut3pYuUUiSA-Vqz5PZ23ncms/view?usp=sharing)  |[stdout.txt](https://drive.google.com/file/d/1EhyPOyHXhdIljNk78byXaXDE2LM8y5j1/view?usp=sharing)
| R101        |Yolact    | 36.5 |
| R101        |CondInst  | 36.5 |
| Swin-tiny   |Yoalct    | 38.0 |
| Swin-tiny   |CondInst  | 39.1 |[cico_CondInst_swint_yt21_f3.pth](https://drive.google.com/file/d/1cH2dK7GxmwcrC4bCKSIB0aF0E8fOXV5_/view?usp=sharing)|[stdout.txt](https://drive.google.com/file/d/1hL6hRbTTH3yG6u2tF98XEY5f72yVq7QW/view?usp=sharing)

### Quantitative Results on OVIS
| Backbone    |FCN       | mAP  | Weights | Results|
|:-----------:|:--------:|:----:|---------|-----------------------------------------------------------------------------------------------------------|
| R50         |Yolact    | 17.2 |
| R50         |CondInst  | 18.0 |
| R101        |Yolact    | 18.7 |
| R101        |CondInst  | 18.2 |
| Swin-tiny   |Yoalct    | 18.0 |
| Swin-tiny   |CondInst  | 18.2 |[cico_CondInst_swint_ovis_f3.pth](https://drive.google.com/file/d/1GEEntoC2or5LKnFPD1z49MuitcP6Xhse/view?usp=sharing)
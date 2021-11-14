# CiCo
An anonymous code repository for our paper submitted in CVPR2022. Unfortunately, Anonymous Github does not support downloading source code now. 
- Clip-in Clip-out: An Alignment-free One-stage Video Instance Segmentation Approach 
 
# Installation
 - Clone this repository 
 - Set up the environment using [Anaconda](https://www.anaconda.com/distribution/):
     - Run `conda env create -f environment.yml`
     - conda activate cico-env
       
 - Install mmcv and compile cocoapi
    - According to your Cuda and pytorch version to install mmcv or mmcv-full from [here](https://github.com/open-mmlab/mmcv). 
    - install cocoapi and a customized COCO API for YouTubeVIS dataset from [here](https://github.com/youtubevos/cocoapi)
      ```Shell
      pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
      git clone https://github.com/youtubevos/cocoapi
      cd cocoapi/PythonAPI
      # To compile and install locally 
      python setup.py build_ext --inplace
      # To install library to Python site-packages 
      python setup.py build_ext install
      ```
    
# Run

## Dataset
If you'd like to train or test CiCo, please download the datasets from the official web: [YTVIS2019](https://youtube-vos.org/dataset/), [YTVIS2021](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/).

## Inference
```Shell
# Display instance masks
python eval.py --config=cico_r50_f3_multiple_yt19.py --trained_model=weights/YTVIS2019/cico_r50_f3_multiple_yt19_e12.pth --display
```

## Training
Make sure to download the entire dataset using the commands above. To train, grab an COCO-pretrained model and put it in `./weights`.
```Shell
# Trains STMask_plus_base_config with a batch_size of 8.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config=configs/CiCo/cico_r50_f3_multiple_yt19.py
```

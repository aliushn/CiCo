# CiCo
An anonymous code repository for our paper submitted in CVPR2022: 
- Clip-in Clip-out: An Alignment-free One-stage Video Instance Segmentation Approach 
 
# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://MinghanLi/CiCo.git
   cd CiCo
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
     - conda activate STMask-env
   - Manually with pip
     - Set up a Python3 environment.
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
       
 - Install mmcv and mmdet
    - According to your Cuda and pytorch version to install mmcv or mmcv-full from [here](https://github.com/open-mmlab/mmcv). Here my cuda and torch version are 10.1 and 1.5.0 respectively. 
      ```Shell
      pip install mmcv-full==1.1.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
      ```
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

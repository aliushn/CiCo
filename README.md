# OSTMask

The code is implmented for our paper submitted in CVPR2021:
 
# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/MinghanLi/STMask.git
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

 - Install spatial-correlation-sampler 
      ```Shell
      pip install spatial-correlation-sampler
      ```
 
 - Complie DCNv2 code (see [Installation](https://github.com/dbolya/yolact#installation))
   - Download code for deformable convolutional layers from [here](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)
     ```Shell
     git clone https://github.com/CharlesShang/DCNv2.git
     cd DCNv2
     python setup.py build develop
     ```

# Dataset
 - If you'd like to train STMask, please download the datasets from the official web: [YTVIS2019](https://youtube-vos.org/dataset/), [YTVIS2021](https://youtube-vos.org/dataset/vis/) and [OVIS](http://songbai.site/ovis/).
 
# Inference
```Shell
# Output a YTVOSEval json to submit to the website.
# This command will create './weights/results.json' for instance segmentation.
python eval.py --config=CiCo_r50_f3_multiple.py --trained_model=weights/STMask_plus_base_ada.pth --mask_det_file=weights/results.json
```

# Training
By default, we train on YouTubeVOS2019 dataset. Make sure to download the entire dataset using the commands above.
 - To train, grab an COCO-pretrained model and put it in `./weights`.
   - [Yolcat++]: For Resnet-50/-101, download `yolact_plus_base_54_80000.pth` or `yolact_plus_resnet_54_80000.pth` from Yolact++ [here](https://github.com/dbolya/yolact).
   - [Yolcat++ & FC]: Alternatively, you can use those Yolact++ with FC models on Table. 2 for training, which can obtain a relative higher performance than that of Yolact++ models.


- Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains STMask_plus_base_config with a batch_size of 8.
CUDA_VISIBLE_DEVICES=0,1 python train.py --config=STMask_plus_base_config --batch_size=8 --lr=1e-4 --save_folder=weights/weights_r101
```

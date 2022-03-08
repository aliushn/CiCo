# CiCo
An offical code repository for our paper submitted in CVPR2022
- Clip-in Clip-out: An Alignment-free One-stage Video Instance Segmentation Approach 

  ![](https://github.com/MinghanLi/CiCo/blob/main/imgs/fifo_cico.png)
 
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
   python eval.py --trained_model=outputs/YTVIS2019/cico_yolact_r50_yt19_f3.pth --NUM_CLIP_FRAMES=3 --overlap_frames=1 
   ```

### Training
   ```Shell
   # Train CiCo with a 3-frame clip on 4 GPUs
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config=configs/CiCo/cico_yolact_r50_yt19.py --NUM_CLIP_FRAMES=3 --IMS_PER_BATCH=6
   ```

## Main Results

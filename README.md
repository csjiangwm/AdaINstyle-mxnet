# Usage 

## Requirements 
* Install skimage via pip: ```sudo pip install scikit-image```
* Install [CUDA](https://developer.nvidia.com/cuda-downloads) and Cudnn
* Install [Mxnet](https://github.com/dmlc/mxnet), please set "USE_NVRTC = 1" in config.mk before compiling
* Download pretrained [VGG model](https://github.com/dmlc/web-data/raw/master/mxnet/neural-style/model/vgg19.params) and save it to the root of this repository.
* Download [MSCOCO](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) dataset if you want to train models.

## Use Trained Models
To use trained models to generate images, first load the model and specify the output shape.
```
python generate.py
```

## Train New Model
First, edit train_gram.py, modify VGGPATH, CONTENTPATH and STYLEPATH
```
python train_gram.py
```

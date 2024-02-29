## Correspondence-based Generative Bayesian Deep Learning forSemi-supervised Volumetric Medical Image Segmentation

A Pytorch implementation of our CMIG 2024 paper "Correspondence-based Generative Bayesian Deep Learning forSemi-supervised Volumetric Medical Image Segmentation".

Thanks
-----
Our code framework is based on the open-source  <a href="https://pan.baidu.com/s/1yOGMBZOzlZ5UJ2EGh9y8CQ">code</a> （CVPR 2022）. Thank you for their open-source code and data.


Build
-----

please run with the following command:

```
conda env create -f requirement.yml
conda activate pytorch
```


Preparation
-----
The datasets can be downloaded from their official sites. We also provide them here:

Baidu Disk: <a href="https://pan.baidu.com/s/1yOGMBZOzlZ5UJ2EGh9y8CQ">download</a>  (code: zr4s)   

Google Drive: <a href="https://drive.google.com/drive/folders/1JprKNLCGQtaCXuVziNHz7HyOMbqzsXrM?usp=sharing">download</a>  

Note that they are saved as 'png', which are extracted from their original datasets without any further preprocessing. 

Plus, please prepare the training and the testing text files. Each file has the following format:

```
/Path/to/the/image/files /Path/to/the/label/map/files
...
...
```
i.e., 'train_AtriaSeg.txt' and 'test_AtriaSeg.txt'
In addition, some path names in train.py need to be modified as necessary.


Training
-----
After the preperation, you can start to train your model. We provide an example file "train_AtriaSeg_16.sh", please run with:

```
sh train_AtriaSeg_16.sh
```
After the training, the latest model will be saved, which is used for testing.

Testing
-----

We provide an example file "test_AtriaSeg_16.sh", please run with:

```
sh test_AtriaSeg_16.sh
```

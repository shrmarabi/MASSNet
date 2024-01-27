# Installation

## Requirements
+ Linux
+ Python 3.5+
+ PyTorch 1.7
+ CUDA 10.0+
+ NCCL 2+
+ GCC 4.9+
+ MMCV 1.7.1 full

#We have tested the following versions of OS and softwares:
+ OS: Ubuntu 18.04.6 LTS 
+ CUDA: 9.2/10.0/11.4
+ NCCL: 2.1.15/2.2.13/2.3.7/2.4.2
+ GCC: 4.9/5.3/5.4/7.3

# Install mmdetection
a. Create a conda virtual environment and activate it. Then install Cython.
 ```
 conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

conda install cython

```
b. Install PyTorch 1.7.0 torchvision 0.8.0.

c. Clone the mmdetection repository.
```
git clone https://github.com/shrmarabi/MASSNet.git
cd MASSNet-main
```

d. Install mmdetection (other dependencies will be installed automatically).
```
python setup.py develop
# or "pip install -v -e ."
```



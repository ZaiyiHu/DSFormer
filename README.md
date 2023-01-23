# DensityToken: Weakly-Supervised Crowd Counting with Density Classification
* This repository is the official implementation of our method DSFormer(Dual Supervision Transformer). In this work, we propose a transformer-based weakly-supervised crowd counting framework with density tokens to perform density classification.

## Overview
![avatar](./framework.jpg)

# Environment

	python >=3.6 
	torch >=1.8.0
	opencv-python >=4.4.0
	scipy >=1.4.0
	h5py >=2.10
	pillow >=7.0.0
	imageio >=1.18
	timm==0.1.30
    tqdm==4.64.0
    grad-cam==1.4.6
- Some crucial packages are listed above. Please make sure to install them before running.
# Datasets
Three datasets are utilized in our proposed method, where links are shown below:
- Download ShanghaiTech dataset from [Baidu-Disk](https://pan.baidu.com/s/15WJ-Mm_B_2lY90uBZbsLwA), passward:cjnx; or [Google-Drive](https://drive.google.com/file/d/1CkYppr_IqR1s6wi53l2gKoGqm7LkJ-Lc/view?usp=sharing)
- Download UCF-QNRF dataset from [here](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- Download NWPU-CROWD dataset from [Baidu-Disk](https://pan.baidu.com/s/1VhFlS5row-ATReskMn5xTw), passward:3awa; or [Google-Drive](https://drive.google.com/file/d/1drjYZW7hp6bQI39u7ffPYwt4Kno9cLu8/view?usp=sharing)




# Training


```
python train.py --dataset ShanghaiA
```


# Testing

You can download the pretrained model from [Baidu-Disk](链接：https://pan.baidu.com/s/1TIqgYdlDp6oa5kF16PKMMg 
), passward:DSFO

```
python test.py --dataset ShanghaiA  --pre model_best.pth
```



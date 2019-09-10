# SDNet

## A Simple and Robust Deep Convolutional Approach to Blind Image Denoising

by [Hengyuan Zhao](https://github.com/zhaohengyuan1/), [Wenze Shao](https://scholar.google.com.hk/citations?hl=zh-CN&user=0iHboRcAAAAJ), [Bingkun Bao](https://scholar.google.com.sg/citations?user=lDppvmoAAAAJ&hl=zh-CN), [Haibo Li](https://scholar.google.com/citations?user=MGZuzNEAAAAJ&hl=en)

## Dependencies

```
Python 3 (Recommend to use Anaconda)

Pytorch >=1.0.0

skimage

h5py

opencv-python

```
## Code

### Datasets

BSD400 was used in paper.

### Training 

The training file in the train documents.

1. clone this github repo.

```
git clone https://github.com/zhaohengyuan1/SDNet.git

cd SDNet
```

2. Generate your training data.(Recommend *.h5 file)

3. Run training file.

```
python ./train/SDN_Color_Blocks3.py
```

### Testing

Our testing codes are wrote in jupyter notebook.

Our Result images were put in Results.

## Network Architecture

![](./figs/All.png)

Figure 1. Illustration of our SDNet for stagewise blind denoising of real photographs.

The total framework has three noise maps as the part of the total noise map and the Block in picture has three strutures in the following picture.

![](./figs/Block.png)

Figure 2. Three distinct building blocks. The left is the plain convolutional block, the middle and right present our lifted residual blocks for the proposed SDNet.

## Results

![](./figs/Nam_table.png)

Table 1. Performance comparison of different blind denoising methods on the real image dataset of Nam et al. [1], including CBM3D [2], DnCNN [3], NC [4], WNNM [5], MCWNNM [6], and our SDNet. The bold indicates the best.


![](./figs/table2.png)

Table 2. Quantitative analysis (average PSNR and SSIM) of the SDNet on two synthetic datasets, i.e., CBSD68 [7] and Kodak, and a real noisy dataset [1]. (a) SDNet only keeping the final noise map, (b) SDNet without utilizing the shortcut connections, (c) SDNet with Block-1 in Figure 2, (d) SDNet with Block-2 in Figure 2, (e) SDNet with Block-3 in Figure 2, i.e., our final blind denoiser. The bold indicates the best.

# Acknowledgement

The study is supported in part by the Natural Science Foundation (NSF) of China (61771250, 61602257, 61572503, 61872424, 61972213, 6193000388), and the NSF of Jiangsu Province (BK20160904).

# Reference

[1] S. Nam, Y. Hwang, Y. Matsushita, S.J. Kim. A holistic approach to cross-channel image noise modeling and its application to image denoising. In CVPR, 2016.

[2] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian. Color image denoising via sparse 3D collaborative filtering with grouping constraint in luminance-chrominance space. In ICIP, 2007.

[3] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. TIP, 26: 3142–3155, 2017.

[4] M. Lebrun, M. Colom, and J.-M. Morel. Multiscale image blind denoising. TIP, 24(10):3149– 3161, 2015.

[5] S. Gu, L. Zhang, W. Zuo, and X. Feng. Weighted nuclear norm minimization with application to image denoising. In CVPR, 2014.

[6] J. Xu, L. Zhang, D. Zhang, and X. Feng. Multi-channel weighted nuclear norm minimization for real color image denoising. In ICCV, 2017.

[7] D. Martin, C. Fowlkes, D. Tal, and J. Malik. A database of human segmented natural images and its application to evaluating segmentation algorithms and measuring ecological statistics. In ICCV, 2001.

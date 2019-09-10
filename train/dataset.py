"""
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata

def data_augmentation(image, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return np.transpose(out, (2, 0, 1))


def img_to_patches(img, win, stride=1):
    r"""Converts an image to an array of patches.

    Args:
        img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
            image
        win: size of the output patches
        stride: int. stride
    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([endc, win*win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return res.reshape([endc, win, win, total_pat_num])

def prepare_data(data_path, \
                 patch_size, \
                 stride, \
                 file_name, \
                 max_num_patches=None, \
                 aug_times=1, \
                 gray_mode=True):
    r"""Builds the training and validations datasets by scanning the
    corresponding directories for images and extracting	patches from them.

    Args:
        data_path: path containing the training image dataset
        val_data_path: path containing the validation image dataset
        patch_size: size of the patches to extract from the images
        stride: size of stride to extract patches
        stride: size of stride to extract patches
        max_num_patches: maximum number of patches to extract
        aug_times: number of times to augment the available data minus one
        gray_mode: build the databases composed of grayscale patches
    """
    # training database
    print('> Training database')
#     scales = [1, 0.9, 0.8, 0.7]
    scales = [1]
    types = ('*.bmp', '*.png', '*.tif', '*.jpg')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(data_path, tp)))
    files.sort()

    if gray_mode:
        traindbf = file_name
    else:
        traindbf = file_name

    if max_num_patches is None:
        max_num_patches = 5000000
        print("\tMaximum number of patches not set")
    else:
        print("\tMaximum number of patches set to {}".format(max_num_patches))
    train_num = 0
    i = 0
    with h5py.File(traindbf, 'w') as h5f:
        while i < len(files) and train_num < max_num_patches:
            imgor = cv2.imread(files[i])
            # h, w, c = img.shape
            for sca in scales:
                img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
                                interpolation=cv2.INTER_CUBIC)
                if not gray_mode:
                    # CxHxW RGB image
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    # CxHxW grayscale image (C=1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)
#                 img = normalize(img)
                patches = img_to_patches(img, win=patch_size, stride=stride)
                print("\tfile: %s scale %.1f # samples: %d" % \
                      (files[i], sca, patches.shape[3]*aug_times))
                for nx in range(patches.shape[3]):
                    data = data_augmentation(patches[:, :, :, nx].copy(), \
                              np.random.randint(0, 7))
                    h5f.create_dataset(str(train_num), data=data)
                    train_num += 1
                    for mx in range(aug_times-1):
                        data_aug = data_augmentation(data, np.random.randint(1, 4))
                        h5f.create_dataset(str(train_num)+"_aug_%d" % (mx+1), data=data_aug)
                        train_num += 1
            i += 1

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num)


class Dataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """
    def __init__(self, train_file=None, train=True, shuffle=True, file_mul=1):
        super(Dataset, self).__init__()
        self.train = train
        self.traindbf = train_file
        self.file_mul = file_mul

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
            
        self.keys = list(h5f.keys())
        if shuffle:
            random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys) // self.file_mul

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File(self.traindbf, 'r')

        key = self.keys[index]
        data = np.array(h5f[key])
        
        h5f.close()
        return data

class SDNDataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """
    def __init__(self, train_file=None, train=True, shuffle=True, file_mul=1):
        super(SDNDataset, self).__init__()
        self.train = train
        self.traindbf = train_file
        self.file_mul = file_mul

        if self.train:
            h5f = h5py.File(self.traindbf, 'r')
            
        self.keys = list(h5f.keys())
        if shuffle:
            random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys) // self.file_mul

    def __getitem__(self, index):
       
        if self.train:
            h5f = h5py.File(self.traindbf, 'r')

        key = self.keys[index]
        data = np.array(h5f[key])
        
        data = data.transpose(1,2,0)

        gamma = np.random.choice([0.4, 0.5])
        sigmaU = np.random.choice(np.arange(0.2, 1, 0.05))
        sigmaW = np.random.randint(5,10)
        u = np.random.normal(0, sigmaU, data.shape)
        w = np.random.normal(0, sigmaW, data.shape)
        noise = np.power(data, gamma)*u + w
        
        dataN = data + noise
        
        data = data.transpose(2,0,1).astype('float32')
        dataN = dataN.transpose(2,0,1).astype('float32')
        
        h5f.close()
        return {'G': data, 'N': dataN}    

from __future__ import print_function
import torch
import os
import numpy as np
from torch.utils.data import Dataset


# Image: [b,c,w,h] = [b,1,64,64]
class Dataset_YH(Dataset):
    def __init__(self, file_dir, small=False, transform=None):
        self.file_dir = file_dir
        self.small = small  # quick test for small dataset
        self.transform = transform
        self.all_img, self.all_spe = self.get_all_data()
        self.len = self.all_spe.shape[0]
        self.im_size = self.all_img.shape[-1]
        self.sp_size = self.all_spe.shape[-1]

    def get_all_data(self):
        img = np.loadtxt(os.path.join(self.file_dir, 'img.txt'))
        spe = np.loadtxt(os.path.join(self.file_dir, 'spe.txt'))
        self.im_size = int(np.sqrt(img.shape[1]))
        img = img.reshape([-1, self.im_size, self.im_size])
        img = torch.from_numpy(img)
        img = img.reshape([-1, 1, self.im_size, self.im_size])
        spe = torch.from_numpy(spe)
        self.sp_size = spe.shape[1] // 2
        spe[:, self.sp_size:] = (spe[:, self.sp_size:] + np.pi) / 2 / np.pi
        spe = torch.stack((spe[:, :self.sp_size], spe[:, self.sp_size:]), dim=2)
        print('amplitude range: (', torch.min(spe[:, :, 0]).item(), ',', torch.max(spe[:, :, 0]).item(), ')')
        print('phase range: (', torch.min(spe[:, :, 1]).item(), ',', torch.max(spe[:, :, 1]).item(), ')')
        if self.small:
            img = img[:1000]
            spe = spe[:1000]
        return img.float(), spe.float()  # img:[batch_size,1,im_size,im_size], spe:[batch_size,sp_size,2]

    def __str__(self):
        return 'YH dataset with [{0},{1},{1}]  img and [{0},{2}] spec'.format(self.len, self.im_size, self.sp_size)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        ## add data augmentation
        img = self.all_img[idx]
        spe = self.all_spe[idx]
        if self.transform is not None:
            img = self.transform(img)
            # img, spe = data_augmentation(img, spe)
            # img = img.reshape([1, self.im_size, self.im_size])  # back to shape 1x64x64
        return img, spe

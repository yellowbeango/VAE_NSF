from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import torchvision.transforms as transforms
import os
import argparse
import sys
import models as models
from utils import mkdir_p, get_mean_and_std
from Datasets import Dataset_YH

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--model', default='VanillaVAE', choices=model_names, type=str, help='choosing network')
parser.add_argument('--latent_dim', default=128, type=int)

# Parameters for  dataset
parser.add_argument('--traindir', default='/home/UNT/jg0737/Desktop/NSFdata/train_data', type=str, metavar='PATH',
                    help='path to training set')
parser.add_argument('--testdir', default='/home/UNT/jg0737/Desktop/NSFdata/test_data', type=str, metavar='PATH',
                    help='path to testing set')

# Parameters for  training
# parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/%s-%s' % (
    args.model, args.latent_dim)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

print('==> Preparing training data..')
train_dataset = Dataset_YH(args.traindir, small=args.small)
test_dataset = Dataset_YH(args.testdir, small=args.small)
print('==> Next')
train_mean, train_std = get_mean_and_std(train_dataset)
test_mean, test_std = get_mean_and_std(test_dataset)
print(f"train mean std: {train_mean} {train_std}")
print(f"test  mean std: {test_mean} {test_std}")

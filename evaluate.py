"""
This file just evaluate the final results.
Target: input a spectrum, and generate an image with almost same spectrum.
"""""

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
import traceback
import models as models
from utils import mkdir_p, get_mean_and_std, Logger, progress_bar, \
    save_model, save_binary_img, plot_spectrum, plot_amplitude
from Datasets import Dataset_YH
from losses import SpectrumLoss

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--forward', default='CNN_res_64_64', choices=model_names, type=str, help='choosing network')
parser.add_argument('--vae', default='VanillaVAE', choices=model_names, type=str, help='choosing vae network')
parser.add_argument('--embedding', default='Embedding',
                    choices=model_names, type=str, help='choosing embedding network, embedding spectrum to mu/var')

parser.add_argument('--forward_resume',
                    default='/home/g1007540910/VAE_NSF/forward2.pt',
                    type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--vae_resume',
                    default='/home/g1007540910/VAE_NSF/checkpoints/VanillaVAE-128/checkpoint_best.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--embedding_resume',
                    default='/home/g1007540910/VAE_NSF/checkpoints/VanillaVAE-128/embedding_best.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint')

parser.add_argument('--latent_dim', default=128, type=int)
# Parameters for  dataset
# parser.add_argument('--traindir', default='/home/g1007540910/NSFdata/train_data', type=str, metavar='PATH',
#                     help='path to training set')
parser.add_argument('--testdir', default='/home/g1007540910/NSFdata/test_data', type=str, metavar='PATH',
                    help='path to testing set')

parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--val_num', default=8, type=int,
                    help=' the number of validate images, also nrow for saving images')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/evaluate/%s-%s-%s' % (
    args.vae, args.forward, args.embedding)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
transform = transforms.Compose([
    SetRange  # rescale to [-1,1] considering the tanh activation
])
# print('==> Preparing training data..')
# train_dataset = Dataset_YH(args.traindir, small=args.small, transform=transform)
# M_N = args.bs / train_dataset.len  # for the loss
print('==> Preparing testing data..')
test_dataset = Dataset_YH(args.testdir, small=args.small, transform=transform)

# data loader
# trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
# testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
valloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_num, shuffle=False, num_workers=4)


def main():
    # Model
    print('==> Building model..')
    forward_net = models.__dict__[args.forward]()
    forward_net = forward_net.to(device)
    vae_net = models.__dict__[args.vae](in_channels=1, latent_dim=args.latent_dim)
    vae_net = vae_net.to(device)
    embedding_net = models.__dict__[args.embedding](points=183, laten_dim=args.latent_dim)
    embedding_net = embedding_net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    print('==> Loading pretrained models..')
    forward_net.load_state_dict(torch.load(args.forward_resume))
    checkpoint = torch.load(args.vae_resume)
    vae_net.load_state_dict(checkpoint['net'])
    checkpoint_embed = torch.load(args.embedding_resume)
    embedding_net.load_state_dict(checkpoint_embed['net'])

    print("===> start evaluating ...")
    generate_images(forward_net, vae_net, embedding_net, valloader, name="test_spectrum.png")
    # sample_images_random(net, name="test_sample_random")
    # sample_images_spectrum(net, name="test_sample_spectrum")


def generate_images(forwardNet, vaeNet, embeddingNet, valloader, name="evaluate"):
    dataloader_iterator = iter(valloader)
    with torch.no_grad():
        img, target = next(dataloader_iterator)
        img = img.to(device)
        target = target.to(device)
        img = img[:args.val_num]
        target = target[:args.val_num]

        embed_mu, embed_var = embeddingNet(target)
        eps = torch.randn_like(embed_mu)
        embed_std = torch.exp(0.5 * embed_var)
        z = eps * embed_std + embed_mu
        predict_img = vaeNet.sample(z)
        predict_spectrum = forwardNet(predict_img)

        img_result = torch.cat([img, predict_img], dim=0)
        save_binary_img(img_result.data,
                        os.path.join(args.checkpoint, f"{name}_imge.png"),
                        nrow=args.val_num)
        plot_amplitude([target, predict_spectrum],
                       ['input', 'generated'],
                       save_path=os.path.join(args.checkpoint, f"{name}_spectrum.png")
                       )


if __name__ == '__main__':
    main()


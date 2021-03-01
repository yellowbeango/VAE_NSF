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
from losses import SpectrumLoss, EmbedLoss, VAELoss

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
                    default='/home/g1007540910/VAE_NSF/checkpoints/VanillaVAE-128/embedding.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint')

parser.add_argument('--latent_dim', default=128, type=int)
# Parameters for  dataset
parser.add_argument('--traindir', default='/home/g1007540910/NSFdata/train_data', type=str, metavar='PATH',
                    help='path to training set')
parser.add_argument('--testdir', default='/home/g1007540910/NSFdata/test_data', type=str, metavar='PATH',
                    help='path to testing set')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--val_num', default=8, type=int,
                    help=' the number of validate images, also nrow for saving images')
# weight for specturm loss
parser.add_argument('--amp_weight', default=1.0, type=float, help='weight for specturm loss')
parser.add_argument('--phi_weight', default=0.1, type=float, help='weight for specturm loss')
parser.add_argument('--bs', default=144, type=int, help='batch size, better to have a square number')


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
train_dataset = Dataset_YH(args.traindir, small=args.small, transform=transform)
M_N = args.bs / train_dataset.len  # for the loss
test_dataset = Dataset_YH(args.testdir, small=args.small, transform=transform)

# data loader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
valloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_num, shuffle=False, num_workers=4)

criterion_forward = SpectrumLoss(amp_weight=args.amp_weight, phi_weight=args.phi_weight)
criterion_embed =EmbedLoss()
criterion_vae = VAELoss(M_N = M_N)

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
    if not args.evaluate:
        print("===> start calculating loss ...")

        train_loss_dict = calculate_loss(forward_net, vae_net, embedding_net, trainloader)
        test_loss_dict = calculate_loss(forward_net, vae_net, embedding_net, testloader)

        file_str = "train_loss_dict\n"
        file_str+=str(train_loss_dict)
        file_str += "\n\n\ntest_loss_dict\n"
        file_str += str(test_loss_dict)

        fh = open(os.path.join(args.checkpoint, "loss.txt"), 'w')
        fh.write(file_str)
        fh.close()






    print("===> start evaluating ...")
    generate_images(forward_net, vae_net, embedding_net, valloader, name="evaluate")



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


def calculate_loss(forwardNet, vaeNet, embeddingNet, loader):
    forward_loss = 0
    forward_loss_amp = 0
    forward_loss_phi = 0

    vae_loss = 0
    vae_recons_loss = 0
    vae_kld_loss = 0

    embed_loss = 0
    embed_mu_loss = 0
    embed_var_loss = 0



    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        forward_result = forwardNet(inputs)
        forward_loss_dict = criterion_forward(forward_result, targets)

        vae_result = vaeNet(inputs)
        vae_loss_dict = criterion_vae(vae_result)  # loss, Reconstruction_Loss, KLD

        vae_mu, vae_var = vae_result["mu"], vae_result["log_var"]
        predict_mu, predict_var = embeddingNet(targets)
        embed_loss_dict = criterion_embed(predict_mu, predict_var, vae_mu, vae_var)


        vae_loss += (vae_loss_dict['loss']).item()
        vae_recons_loss += (vae_loss_dict['Reconstruction_Loss']).item()
        vae_kld_loss += (vae_loss_dict['KLD']).item()


        forward_loss += (forward_loss_dict['loss']).item()
        forward_loss_amp += (forward_loss_dict['loss_amp']).item()
        forward_loss_phi += (forward_loss_dict['loss_phi']).item()

        embed_loss += (embed_loss_dict['loss']).item()
        embed_mu_loss += (embed_loss_dict['mu_Loss']).item()
        embed_var_loss += (embed_loss_dict['var_loss']).item()

        progress_bar(batch_idx, len(trainloader), 'Forward_loss: %.3f | vae_Loss: %.3f | embed_Loss: %.3f'
                     % (forward_loss / (batch_idx + 1), vae_loss / (batch_idx + 1), embed_loss / (batch_idx + 1)))

    return {
        "forward_loss": forward_loss / (batch_idx + 1),
        "forward_loss_amp": forward_loss_amp / (batch_idx + 1),
        "forward_loss_phi": forward_loss_phi / (batch_idx + 1),
        "vae_loss": vae_loss / (batch_idx + 1),
        "vae_recons_loss": vae_recons_loss / (batch_idx + 1),
        "vae_kld_loss": vae_kld_loss / (batch_idx + 1),
        "embed_loss": embed_loss / (batch_idx + 1),
        "embed_mu_loss": embed_mu_loss / (batch_idx + 1),
        "embed_var_loss": embed_var_loss / (batch_idx + 1)
    }


if __name__ == '__main__':
    main()


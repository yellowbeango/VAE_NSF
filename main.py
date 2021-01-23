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
from utils import mkdir_p, get_mean_and_std, Logger, progress_bar, save_model
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
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=144, type=int, help='batch size, better to have a square number')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--scheduler_gamma', default=0.95, type=float, help='weight decay')

parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/%s-%s' % (
    args.model, args.latent_dim)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

print('==> Preparing training data..')
train_dataset = Dataset_YH(args.traindir, small=args.small)
M_N = args.bs/train_dataset.len  # for the loss
print('==> Preparing testing data..')
test_dataset = Dataset_YH(args.testdir, small=args.small)
# train_mean, train_std = get_mean_and_std(train_dataset)
# test_mean, test_std = get_mean_and_std(test_dataset)
# print(f"train mean std: {train_mean} {train_std}")
# print(f"test  mean std: {test_mean} {test_std}")
# train mean std: 0.27979007363319397 0.44508227705955505
# test  mean std: 0.2811497151851654 0.44574955105781555
transform_train = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2798,), (0.4451,)),
])
# data loader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)


def main():
    print("==> start training..")
    start_epoch = 0

    # Model
    print('==> Building model..')
    net = models.__dict__[args.model](in_channels=1, latent_dim=args.latent_dim)
    net = net.to(device)

    if device == 'cuda':
        # Considering the data scale and model, it is unnecessary to use DistributedDataParallel
        # which could speed up the training and inference compared to DataParallel
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Recons Loss', 'KLD Loss'])


    if not args.evaluate:
        # training
        for epoch in range(start_epoch, args.es):
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, scheduler.get_last_lr()[-1]))
            train_out = train(net, trainloader, optimizer)  # {train_loss, recons_loss, kld_loss}
            save_model(net, epoch, os.path.join(args.checkpoint, 'checkpoint.pth'))
            logger.append([epoch + 1, scheduler.get_last_lr()[-1],
                           train_out["train_loss"], train_out["recons_loss"], train_out["kld_loss"]])
            scheduler.step()
        logger.close()
        print(f"\n==> Finish training..\n")

    print("===> Evaluating （TODO） ...")


def train(net, trainloader, optimizer):
    net.train()
    train_loss = 0
    recons_loss = 0
    kld_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        optimizer.zero_grad()
        result = net(inputs)
        loss_dict = net.module.loss_function(*result, M_N=M_N) # loss, Reconstruction_Loss, KLD
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recons_loss += (loss_dict['Reconstruction_Loss']).item()
        kld_loss += (loss_dict['KLD']).item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Rec_Loss: %.3f | KLD_Loss: %.3f'
                     % (train_loss / (batch_idx + 1), recons_loss / (batch_idx + 1), kld_loss / (batch_idx + 1)))

    return {
        "train_loss": train_loss / (batch_idx + 1),
        "recons_loss": recons_loss / (batch_idx + 1),
        "kld_loss": kld_loss / (batch_idx + 1)
    }


if __name__ == '__main__':
    main()

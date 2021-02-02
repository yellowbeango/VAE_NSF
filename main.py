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
from utils import mkdir_p, imbinarize, Logger, progress_bar, save_model, save_binary_img, merge, convert_ypred, forward_loss
from Datasets import Dataset_YH
from models.model_forward import CNN_res_64_64

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--model', default='VanillaVAE', choices=model_names, type=str, help='choosing network')
parser.add_argument('--latent_dim', default=64, type=int)

# Parameters for  dataset
parser.add_argument('--traindir', default='../Hologram-Yihao-CST/data/train_data', type=str, metavar='PATH',
                    help='path to training set')
parser.add_argument('--testdir', default='../Hologram-Yihao-CST/data/test_data', type=str, metavar='PATH',
                    help='path to testing set')

# Parameters for  training
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--es', default=50, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=64, type=int, help='batch size, better to have a square number')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--scheduler_gamma', default=0.95, type=float, help='weight decay')

parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')
parser.add_argument('--val_num', default=8, type=int,
                    help=' the number of validate images, also nrow for saving images')

args = parser.parse_args()


def main():
    start_epoch = 0
    best_loss = 9999999.99

    # Model
    print('==> Building model..')
    for_net = CNN_res_64_64().to(device)
    for_net.load_state_dict(torch.load('./checkpoints/ampphase.pt'))
    net = models.__dict__[args.model](in_channels=1, latent_dim=args.latent_dim, spec_dim=366)
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
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
            print('==> Resuming from checkpoint, loaded..')
        else:
            print("==> No checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Recons Loss', 'KLD Loss', 'Pred Loss'])

    if not args.evaluate:
        # training
        print("==> start training..")
        for epoch in range(start_epoch, args.es):
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, scheduler.get_last_lr()[-1]))
            train_out = train(net, for_net, trainloader, optimizer)  # {train_loss, recons_loss, kld_loss}
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint.pth'))
            if train_out["train_loss"] < best_loss:
                save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint_best.pth'),
                           loss=train_out["train_loss"])
                best_loss = train_out["train_loss"]
            logger.append([epoch + 1, scheduler.get_last_lr()[-1],
                           train_out["train_loss"], train_out["recons_loss"], train_out["kld_loss"], train_out["pred_loss"]])
            scheduler.step()
        logger.close()
        print(f"\n==> Finish training..\n")

    print("===> start evaluating ...")
    generate_images(net, valloader, name="test_reconstruct")
    sample_images(net, valloader, name="test_randsample")


def train(net, for_net, trainloader, optimizer):
    net.train()
    train_loss = 0
    recons_loss = 0
    kld_loss = 0
    pred_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        result = net(inputs, targets)
        loss_dict = net.module.loss_function(*result, M_N=M_N)  # loss, Reconstruction_Loss, KLD
        recon = result[0]   # retrieved image [-1,1]
        recon = imbinarize(recon) # binarize image 0 or 1
        pred_target = convert_ypred(for_net(recon))
        p_loss = forward_loss(pred_target, targets)

        loss = loss_dict['loss'] + 10 * p_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        recons_loss += (loss_dict['Reconstruction_Loss']).item()
        kld_loss += (loss_dict['KLD']).item()
        pred_loss += p_loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Rec_Loss: %.3f | KLD_Loss: %.3f | Pred_Loss: %.3f'
                     % (train_loss / (batch_idx + 1), recons_loss / (batch_idx + 1), kld_loss / (batch_idx + 1), pred_loss / (batch_idx + 1)))

    return {
        "train_loss": train_loss / (batch_idx + 1),
        "recons_loss": recons_loss / (batch_idx + 1),
        "kld_loss": kld_loss / (batch_idx + 1),
        "pred_loss": pred_loss / (batch_idx + 1)
    }


def generate_images(net, valloader, name="val"):
    dataloader_iterator = iter(valloader)
    with torch.no_grad():
        img, spe = next(dataloader_iterator)
        img = img.to(device)
        spe = spe.to(device)
        recons = net.module.generate(img, spe)
        result = torch.cat([img, recons], dim=0)
        save_binary_img(result.data,
                        os.path.join(args.checkpoint, f"{name}.png"),
                        nrow=args.val_num)


def sample_images(net, valloader, name="val"):
    dataloader_iterator = iter(valloader)
    with torch.no_grad():
        img, spe = next(dataloader_iterator)
        spe = spe.to(device)
        z = torch.randn(args.val_num, args.latent_dim)
        z = z.to(device)
        sampled = net.module.sample(z, spe)
        save_binary_img(sampled.data,
                        os.path.join(args.checkpoint, f"{name}.png"),
                        nrow=args.val_num)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.checkpoint = './checkpoints/%s-%s' % (
        args.model, args.latent_dim)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    SetRange = transforms.Lambda(merge)
    # SetRangedoesn't work on Windows
    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        SetRange  # rescale to [-1,1] considering the tanh activation
    ])
    print('==> Preparing training data..')
    train_dataset = Dataset_YH(args.traindir, small=args.small, transform=transform)
    M_N = args.bs / train_dataset.len  # for the loss
    print('==> Preparing testing data..')
    test_dataset = Dataset_YH(args.testdir, small=args.small, transform=transform)

    # train_mean, train_std = get_mean_and_std(train_dataset)
    # test_mean, test_std = get_mean_and_std(test_dataset)
    # print(f"train mean std: {train_mean} {train_std}")
    # print(f"test  mean std: {test_mean} {test_std}")
    # train mean std: 0.27979007363319397 0.44508227705955505
    # test  mean std: 0.2811497151851654 0.44574955105781555

    # data loader
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
    valloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_num, shuffle=False, num_workers=4)
    main()

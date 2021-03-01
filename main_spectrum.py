"""
This file just train a spectrum decoder, nothing related to VAE.
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
    save_model, save_binary_img, plot_spectrum, plot_amplitude, plot_amplitude_and_phi
from Datasets import Dataset_YH
from losses import SpectrumLoss

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--model', default='CNN_res_64_64', choices=model_names, type=str, help='choosing network')

# Parameters for  dataset
parser.add_argument('--traindir', default='/home/g1007540910/NSFdata/train_data', type=str, metavar='PATH',
                    help='path to training set')
parser.add_argument('--testdir', default='/home/g1007540910/NSFdata/test_data', type=str, metavar='PATH',
                    help='path to testing set')

# Parameters for  training
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=256, type=int, help='batch size, better to have a square number')
parser.add_argument('--scheduler_gamma', default=0.5, type=float, help='scheduler reduce')
parser.add_argument('--scheduler_step', default=5, type=int, help='scheduler step')
# weight for specturm loss
parser.add_argument('--amp_weight', default=1.0, type=float, help='weight for specturm loss')
parser.add_argument('--phi_weight', default=0.1, type=float, help='weight for specturm loss')

parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')
parser.add_argument('--val_num', default=8, type=int,
                    help=' the number of validate images, also nrow for saving images')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/spectrumEncoder/%s-amp%s-phi%s' % (
    args.model, args.amp_weight, args.phi_weight)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
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

# data loader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
valloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_num, shuffle=False, num_workers=4)


def main():
    start_epoch = 0
    best_loss = 9999999.99

    # Model
    print('==> Building model..')
    net = models.__dict__[args.model]()
    net = net.to(device)

    if device == 'cuda':
        # Considering the data scale and model, it is unnecessary to use DistributedDataParallel
        # which could speed up the training and inference compared to DataParallel
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    criterion = SpectrumLoss(args.amp_weight, args.phi_weight)

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            try:
                net.load_state_dict(checkpoint)
            except:
                net.load_state_dict(checkpoint['net'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
            print('==> Resuming from checkpoint, loaded..')
        else:
            print("==> No checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch Number', 'Learning Rate', 'Train Loss', "amp Loss", "phi Loss"])

    if not args.evaluate:
        # training
        print("==> start training..")
        for epoch in range(start_epoch, args.es):
            current_lr = optimizer.param_groups[0]['lr']
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, current_lr))
            train_out = train(net, trainloader, optimizer, criterion)  # {train_loss, recons_loss, kld_loss}
            save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint.pth'))
            if train_out["train_loss"] < best_loss:
                save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'checkpoint_best.pth'),
                           loss=train_out["train_loss"])
                best_loss = train_out["train_loss"]
            logger.append([epoch + 1, current_lr, train_out["train_loss"],
                           train_out["loss_amp"], train_out["loss_phi"]
                           ])
            scheduler.step()
            if epoch % 5 == 0:
                generate_images(net, valloader, name=f"epoch{epoch}_spectrum.png")
        logger.close()
        print(f"\n==> Finish training..\n")

    print("===> start evaluating ...")
    generate_images(net, valloader, name="test_spectrum.png")
    # sample_images_random(net, name="test_sample_random")
    # sample_images_spectrum(net, name="test_sample_spectrum")


def train(net, trainloader, optimizer, criterion):
    net.train()
    train_loss = 0
    loss_amp = 0
    loss_phi = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        results = net(inputs)
        loss_dict = criterion(results, targets)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loss_amp += (loss_dict['loss_amp']).item()
        loss_phi += (loss_dict['loss_phi']).item()

        progress_bar(batch_idx, len(trainloader),
                     'All:%.3f |amp:%.3f |hpi:%.3f'
                     % (train_loss / (batch_idx + 1),
                        loss_amp / (batch_idx + 1),
                        loss_phi / (batch_idx + 1)
                        )
                     )

    return {
        "train_loss": train_loss / (batch_idx + 1),
        "loss_amp": loss_amp / (batch_idx + 1),
        "loss_phi": loss_phi / (batch_idx + 1)
    }


def generate_images(net, valloader, name="val.png"):
    dataloader_iterator = iter(valloader)
    with torch.no_grad():
        img, target = next(dataloader_iterator)
        img = img.to(device)
        target = target.to(device)
        img = img[:args.val_num]
        target = target[:args.val_num]
        predict = net(img)
        plot_amplitude_and_phi([target, predict],
                               ['target', 'predict'],
                               num=args.val_num,
                               save_path=os.path.join(args.checkpoint, f"{name}")
                               )


if __name__ == '__main__':
    main()
    # try:
    #     main()
    # except Exception as e:
    #     print(e)
    #     traceback.print_exc()
    #     os.system("sudo poweroff")
    # print("DONE, FINISHED!!!")
    # os.system("sudo poweroff")

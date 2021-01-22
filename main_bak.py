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
from utils import mkdir_p

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='VAE training for NSF project')


# General MODEL parameters
parser.add_argument('--model', default='VanillaVAE', choices=model_names, type=str, help='choosing network')
parser.add_argument('--latent_dim', default=128, type=int)

# Parameters for  training
# parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
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


print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = MNIST(root='../../data', train=True, download=True, transform=transform,
                 train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                 includes_all_train_class=args.includes_all_train_class)
testset = MNIST(root='../../data', train=False, download=True, transform=transform,
                train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                includes_all_train_class=args.includes_all_train_class)
# data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.stage1_bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.stage1_bs, shuffle=False, num_workers=4)


def main():
    print(device)
    stage1_dict = main_stage1()


def main_stage1():
    print(f"\nStart Stage-1 training ...\n")
    # for  initializing backbone, two branches, and centroids.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Model
    print('==> Building model..')
    net = DFPNet(backbone=args.arch, num_classes=args.train_class_num, embed_dim=args.embed_dim)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.stage1_resume:
        # Load checkpoint.
        if os.path.isfile(args.stage1_resume):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.stage1_resume)
            net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']
            logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'), resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log_stage1.txt'))
        logger.set_names(['Epoch', 'Train Loss', 'Train Acc.'])

    # after resume
    criterion = DFPLoss(temperature=args.temperature)
    optimizer = optim.SGD(net.parameters(), lr=args.stage1_lr, momentum=0.9, weight_decay=5e-4)
    if not args.evaluate:
        for epoch in range(start_epoch, args.stage1_es):
            adjust_learning_rate(optimizer, epoch, args.stage1_lr, factor=args.stage1_lr_factor, step=args.stage1_lr_step)
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, optimizer.param_groups[0]['lr']))
            train_out = stage1_train(net, trainloader, optimizer, criterion, device)
            save_model(net, epoch, os.path.join(args.checkpoint, 'stage_1_last_model.pth'))
            logger.append([epoch + 1, train_out["train_loss"], train_out["accuracy"]])
            if args.plot:
                plot_feature(net, args, trainloader, device, args.plotfolder, epoch=epoch,
                             plot_class_num=args.train_class_num, plot_quality=args.plot_quality)
                plot_feature(net, args, testloader, device, args.plotfolder, epoch="test" + str(epoch),
                             plot_class_num=args.train_class_num + 1, plot_quality=args.plot_quality, testmode=True)
        logger.close()
        print(f"\nFinish Stage-1 training...\n")

    print("===> Evaluating ...")
    stage1_test(net, testloader, device)

    return {
        "net": net
    }


# Training
def stage1_train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    # loss_classification = 0
    # loss_distance = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        out = net(inputs)
        loss_dict = criterion(out, targets)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # loss_classification += (loss_dict['classification']).item()
        # loss_distance += (loss_dict['distance']).item()

        _, predicted = (out['normweight_fea2cen']).max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    return {
        "train_loss": train_loss / (batch_idx + 1),
        # "loss_classification": loss_classification / (batch_idx + 1),
        # "loss_distance": loss_distance / (batch_idx + 1),
        "accuracy": correct / total
    }


def stage1_test(net, testloader, device):
    correct = 0
    total = 0
    norm_fea_list, normweight_fea2cen_list, cosine_fea2cen_list,softmax_list = [], [], [], []
    Target_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            out = net(inputs) # shape [batch,class]
            # energy = (out["normweight_fea2cen"]).sum(dim=1, keepdim=False)
            # energy = torch.logsumexp(out["normweight_fea2cen"], dim=1, keepdim=False)
            norm_fea_list.append(out["norm_fea"])
            normweight_fea2cen_list.append(out["normweight_fea2cen"])
            cosine_fea2cen_list.append(out["cosine_fea2cen"])
            softmax_list.append((out["cosine_fea2cen"].softmax(dim=1).max(dim=1,keepdim=False))[0])
            Target_list.append(targets)

            _, predicted = (out["normweight_fea2cen"]).max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), '| Acc: %.3f%% (%d/%d)'
                         % (100. * correct / total, correct, total))

    print("\nTesting results is {:.2f}%".format(100. * correct / total))

    norm_fea_list = torch.cat(norm_fea_list, dim=0)
    normweight_fea2cen_list = torch.cat(normweight_fea2cen_list, dim=0)
    cosine_fea2cen_list = torch.cat(cosine_fea2cen_list, dim=0)
    softmax_list = torch.cat(softmax_list, dim=0)
    Target_list = torch.cat(Target_list, dim=0)

    energy_hist(norm_fea_list, Target_list, args, "norm")
    energy_hist(torch.logsumexp(norm_fea_list, dim=1, keepdim=False), Target_list, args, "norm_energy")

    energy_hist(normweight_fea2cen_list, Target_list, args, "normweight")
    energy_hist(torch.logsumexp(normweight_fea2cen_list, dim=1, keepdim=False), Target_list, args, "normweight_energy")

    energy_hist(cosine_fea2cen_list, Target_list, args, "cosine")
    energy_hist(torch.logsumexp(cosine_fea2cen_list, dim=1, keepdim=False), Target_list, args, "cosine_energy")

    energy_hist(softmax_list, Target_list, args, "softmax")

    # # Energy analysis
    # Energy_list = torch.cat(Energy_list, dim=0)
    # Target_list = torch.cat(Target_list, dim=0)
    # unknown_label = Target_list.max()
    # unknown_Energy_list = Energy_list[Target_list == unknown_label]
    # known_Energy_list = Energy_list[Target_list != unknown_label]
    # unknown_hist = torch.histc(unknown_Energy_list, bins=args.hist_bins, min=Energy_list.min().data,
    #                            max=Energy_list.max().data)
    # known_hist = torch.histc(known_Energy_list, bins=args.hist_bins, min=Energy_list.min().data,
    #                            max=Energy_list.max().data)
    # print(f"unknown_hist: \n{unknown_hist}")
    # print(f"known_hist: \n{known_hist}")



def save_model(net, epoch, path, **kwargs):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
    }
    for key, value in kwargs.items():
        state[key] = value
    torch.save(state, path)


if __name__ == '__main__':
    main()

from __future__ import print_function

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
import models as models
from utils import mkdir_p, Logger, progress_bar, save_model, save_binary_img
from Datasets import Dataset_YH
from losses import EmbedLoss

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='VAE training for NSF project')

# General MODEL parameters
parser.add_argument('--model', default='VanillaVAE', choices=model_names, type=str, help='choosing network')
parser.add_argument('--latent_dim', default=128, type=int)
parser.add_argument('--embed', default='Embedding', choices=model_names, type=str, help='choosing embedding network')

# Parameters for  dataset
parser.add_argument('--traindir', default='/home/g1007540910/NSFdata/train_data', type=str, metavar='PATH',
                    help='path to training set')
parser.add_argument('--testdir', default='/home/g1007540910/NSFdata/test_data', type=str, metavar='PATH',
                    help='path to testing set')

# Parameters for  training
parser.add_argument('--resume_vae', default='/home/g1007540910/VAE_NSF/checkpoints/VanillaVAE-128/checkpoint_best.pth',
                    type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--resume', default='',
                    type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('--small', action='store_true', help='Showcase on small set')
parser.add_argument('--es', default=80, type=int, help='epoch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=256, type=int, help='batch size, better to have a square number')
parser.add_argument('--wd', default=0.0, type=float, help='weight decay')
parser.add_argument('--scheduler_gamma', default=0.95, type=float)

parser.add_argument('--evaluate', action='store_true', help='Evaluate model, ensuring the resume path is given')
parser.add_argument('--val_num', default=8, type=int,
                    help=' the number of validate images, also nrow for saving images')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.checkpoint = './checkpoints/%s-%s' % (
    args.embed, args.latent_dim)
if not os.path.isdir(args.checkpoint):
    mkdir_p(args.checkpoint)

SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
transform = transforms.Compose([SetRange])
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
    print('==> Building VAE model..')
    net = models.__dict__[args.model](in_channels=1, latent_dim=args.latent_dim)
    net = net.to(device)
    print('==> Loading pretrained VAE model..')
    if os.path.isfile(args.resume_vae):
        checkpoint = torch.load(args.resume_vae)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded VAE from checkpoint, loaded..')
    else:
        print("==> No VAE checkpoint found at '{}'".format(args.resume))

    print('==> Building Embedding model..')
    embed = models.__dict__[args.embed](points=183, laten_dim=args.latent_dim)
    embed = embed.to(device)

    if device == 'cuda':
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    optimizer = optim.Adam(embed.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.scheduler_gamma)
    criterion = EmbedLoss()

    if args.resume:
        # Load checkpoint.
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            embed.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), resume=True)
            print('==> Resuming from checkpoint, loaded..')
        else:
            print("==> No checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'MU Loss', 'VAR Loss'])

    if not args.evaluate:
        # training
        print("==> start training..")
        for epoch in range(start_epoch, args.es):
            print('\nStage_1 Epoch: %d | Learning rate: %f ' % (epoch + 1, scheduler.get_last_lr()[-1]))
            train_out = train(net, embed, trainloader, optimizer,criterion)  # {train_loss, recons_loss, kld_loss}
            save_model(embed, optimizer, epoch, os.path.join(args.checkpoint, 'embedding.pth'))
            if train_out["train_loss"] < best_loss:
                save_model(net, optimizer, epoch, os.path.join(args.checkpoint, 'embedding_best.pth'),
                           loss=train_out["train_loss"])
                best_loss = train_out["train_loss"]
            logger.append([epoch + 1, scheduler.get_last_lr()[-1],
                           train_out["train_loss"], train_out["mu_loss"], train_out["var_loss"]])
            scheduler.step()
        logger.close()
        print(f"\n==> Finish training..\n")


def train(net, embed, trainloader, optimizer, criterion):
    embed.train()
    train_loss = 0
    mu_loss = 0
    var_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            result = net(inputs)
        vae_mu, vae_var = result["mu"], result["log_var"]
        predict_mu, predict_var = embed(targets)
        loss_dict = criterion(predict_mu, predict_var, vae_mu, vae_var)
        loss = loss_dict['loss']
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        mu_loss += (loss_dict['mu_Loss']).item()
        var_loss += (loss_dict['var_loss']).item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | mu_Loss: %.3f | var_Loss: %.3f'
                     % (train_loss / (batch_idx + 1), mu_loss / (batch_idx + 1), var_loss / (batch_idx + 1)))

    return {
        "train_loss": train_loss / (batch_idx + 1),
        "mu_loss": mu_loss / (batch_idx + 1),
        "var_loss": var_loss / (batch_idx + 1)
    }


if __name__ == '__main__':
    main()

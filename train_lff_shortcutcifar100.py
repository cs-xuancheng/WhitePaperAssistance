import argparse
import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from resnet import ResNet
from ShortcutCIFAR100 import ShortcutCIFAR100, CIFAR99

# Parse arguments
parser = argparse.ArgumentParser(description='LfF on Shortcut-CIFAR100')
# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 128)')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# NetWork
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--block_name', type=str, default='bottleneck',
                    help='bottleneck/basicblock, define the building block of ResNet here')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0                    # best test accuracy
best_epoch = 0                  # best epoch


def main():
    global best_acc, best_epoch
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing dataset Shortcut-CIFAR100')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    global num_classes
    num_classes = 100
    path = '../data/cifar-100/'

    # Shortcut-CIFAR100
    trainset = ShortcutCIFAR100(root=path, train=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testset = ShortcutCIFAR100(root=path, train=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # CIFAR99
    testset_cifar99 = CIFAR99(root=path, transform=transform_test)
    testloader_cifar99 = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    global m_times
    m_times = len(trainloader)  # For the sake of simplicity, we usually set M equal to the length of the dataloader.

    print('==> Preparing Model')
    model_b = ResNet(depth=56, num_classes=num_classes, block_name=args.block_name)
    model_d = ResNet(depth=56, num_classes=num_classes, block_name=args.block_name)

    if use_cuda:
        model_b = torch.nn.DataParallel(model_b).cuda()
        model_d = torch.nn.DataParallel(model_d).cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model_d.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_b = optim.SGD(model_b.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer_b, epoch)
        adjust_learning_rate(optimizer_d, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train(trainloader, model_b, model_d, criterion, optimizer_b, optimizer_d, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model_d, criterion, epoch, use_cuda)
        _, test_acc_2 = test(testloader_cifar99, model_d, criterion, epoch, use_cuda)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best is True:
            best_epoch = epoch + 1
            best_cifar99_acc = test_acc_2

    print('Best acc| Shortcut-CIFAR100:{}  | CIFAR99:{} | Epoch:{}'.format(best_acc, best_cifar99_acc, best_epoch))


def train(train_loader, model_b, model_d, criterion, optimizer_b, optimizer_d, epoch, use_cuda):
    # switch to train mode
    model_b.train()
    model_d.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs_b = model_b(inputs)
        outputs_d = model_d(inputs)
        loss_weight = criterion(outputs_b, targets).cpu().detach() / (criterion(outputs_b, targets).cpu().detach() +
                                                                      criterion(outputs_d,targets).cpu().detach() +
                                                                      1e-8)
        loss_d_update = criterion(outputs_d, targets) * loss_weight.cuda()
        loss_b_update = GeneralizedCELoss()(outputs_b, targets)
        loss = loss_b_update + loss_d_update

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs_d.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()

        if (batch_idx+1) % args.print_freq == 0:
            print('\rReal Images Training | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
                epoch+1, args.epochs, batch_idx+1, len(train_loader), losses.avg, top1.avg, top5.avg), end='', flush=True)

    print('\rReal Images Training | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
        epoch + 1, args.epochs, batch_idx+1, len(train_loader), losses.avg, top1.avg, top5.avg), end='\n')
    return losses.avg, top1.avg


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            if (batch_idx+1) % 100 == 0:
                print('\rTesting              | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} | Top-1:{:.2f} | Top-5:{:.2f}'.format(
                    epoch + 1, args.epochs, batch_idx+1, len(val_loader), losses.avg, top1.avg, top5.avg),
                    end='',  flush=True)
    print()
    return losses.avg, top1.avg


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class GeneralizedCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight

        return loss.mean()


if __name__ == '__main__':
    main()


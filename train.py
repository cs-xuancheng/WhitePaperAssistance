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

# Parse arguments
parser = argparse.ArgumentParser(description='WhitePaperAssistance-Pytorch')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar100', type=str, help='cifar10/cifar100')
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
# White Paper Parameter
parser.add_argument('--trigger', type=float, default=1, help='probablity to perform White Paper Assistance')
parser.add_argument('--lambda_para', type=float, default=1.0, help='strength of White Paper Assistance')

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
    print('==> Preparing dataset %s' % args.dataset)
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
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
        path = '../data/cifar-10'
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100
        path = '../data/cifar-100/'
    trainset = dataloader(root=path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testset = dataloader(root=path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    global m_times
    m_times = len(trainloader)  # For the sake of simplicity, we usually set M equal to the length of the dataloader.

    print('==> Preparing Model')
    model = ResNet(depth=args.depth, num_classes=num_classes, block_name=args.block_name)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)

        # White Paper Assistance
        random_starter = random.random()
        if random_starter < args.trigger:
            white_paper_train(model, optimizer, epoch)

            # To demonstrate the performance drop after WP. It has nothing to do with training and can be safely deleted
            test(testloader, model, criterion, epoch, use_cuda)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if is_best is True:
            best_epoch = epoch + 1

    print('Best acc:')
    print(best_acc)
    print('Best Epoch:')
    print(best_epoch)


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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


def white_paper_train(model, optimizer, epoch):
    # switch to train mode
    model.train()
    losses = AverageMeter()

    for i in range(m_times):
        # the easiest way to generate white paper
        white_paper_gen = torch.ones(args.train_batch, 3, 32, 32)
        # white_paper_gen = generate_normalized_white_paper()
        white_paper_gen = white_paper_gen.cuda()
        outputs_wp = model(white_paper_gen)
        outputs_wp_softmax = F.softmax(outputs_wp, dim=-1)

        white_result = (1/num_classes) * torch.ones(args.train_batch, num_classes).cuda()
        loss = args.lambda_para * F.kl_div(F.log_softmax(outputs_wp_softmax, dim=-1),
                                       F.softmax(white_result, dim=-1), reduction='batchmean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss, args.train_batch)

        if (i+1) % args.print_freq == 0:
            print('\rWhite Paper Training | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} |'.format(
                    epoch + 1, args.epochs, i+1, m_times, losses.avg), end='',  flush=True)
    print('\rWhite Paper Training | Epoch:{}/{}| Batch: {}/{}| Losses:{:.4f} |'.format(
                    epoch + 1, args.epochs, i+1, m_times, losses.avg), end='\n')


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


def generate_normalized_white_paper():
    """
    Generate a batch of white paper, and normalize them. The performance using normalization is similar to
    that without normalizing. In this paper, we adopt the simplest version as in line 226.
    """
    white_paper_gen = 255 * np.ones((32, 32, 3), dtype=np.uint8)
    white_paper_gen = Image.fromarray(white_paper_gen)
    white_paper_gen = transforms.ToTensor()(white_paper_gen)
    white_paper_gen = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(white_paper_gen)
    # generate a batch of white papers
    batch_white_paper_gen = torch.ones(args.train_batch, 3, 32, 32)
    white_paper_final = torch.mul(batch_white_paper_gen, white_paper_gen)
    return white_paper_final


if __name__ == '__main__':
    main()


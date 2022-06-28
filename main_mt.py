from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loss_functions import *
from attacks import *
from resnet import ResNet18
from utils import Bar, Logger, AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Adversarial Training')
# Datasets
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=str, default='at',choices=['trades','cosine','at'],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--loss_type',default='trades',type=str)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# attack
parser.add_argument('--epsilon', type=float, default=8./255, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=2./255, help='step size')

# mean teacher
parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
parser.add_argument('--consistency',type=float,default=30)
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--start_ema', default=100, type=int,
                    help='start epoch of self-adaptive training (default 100)')
parser.add_argument('--end_ema', default=150, type=int,
                    help='end epoch of self-adaptive training (default 150)')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed',default=1234)

#Device options
parser.add_argument('--gpu-id', default='0,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device("cuda")

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = datasets.CIFAR10(root='../dataset', train=True, download=False, transform=transform_train)
testset = datasets.CIFAR10(root='../dataset', train=False, download=False, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers,pin_memory=True)
testloader = DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers,pin_memory=True)
best_acc = 0  # best test accuracy
global_step = 0
def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    save_path = args.checkpoint
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = ResNet18()
    ema_model = ResNet18()
    model = torch.nn.DataParallel(model).to(device)
    ema_model = torch.nn.DataParallel(ema_model).to(device)
    for param in ema_model.parameters():
        param.detach_()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.schedule == 'at':
        milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
        scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma)
    elif args.schedule == 'trades':
        milestones = [int(args.epochs * 0.75), int(args.epochs * 0.9)]
        scheduler = MultiStepLR(optimizer,milestones=milestones,gamma=args.gamma)
    elif args.schedule == 'cosine':
        scheduler = CosineAnnealingLR(optimizer,T_max=args.epochs)
    else:
        raise NotImplementedError
    # Resume
    title = 'cifar-10-resnet-mt'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        save_path = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['schedule'])
        global_step = checkpoint['global_step']
        logger = Logger(os.path.join(save_path, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(save_path, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Train Natural Acc.', 'Train Adv Acc.','Valid Natural Acc.','Valid adv Acc.'])
    
    criterion = mean_teacher(args.start_ema,consistency_type=args.consistency_type)
    # criterion = trades_teacher(args.start_ema,consistency_type=args.consistency_type,beta=6.0)
    
    # Train and val
    for epoch in range(start_epoch, args.epochs+1):
        cur_lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, cur_lr))
        weight = get_current_consistency_weight(epoch)
        train_loss,global_step = train(trainloader, criterion,model,ema_model,weight, optimizer, epoch)
        scheduler.step()
        if epoch >= args.start_ema:
            train_acc_x, train_acc_adv = eval_train(trainloader, ema_model)
            test_acc_x, test_acc_adv = eval_test(testloader, ema_model)
        else:
            train_acc_x, train_acc_adv = eval_train(trainloader, model)
            test_acc_x, test_acc_adv = eval_test(testloader, model)
        # append logger file
        logger.append([cur_lr, train_loss, train_acc_x, train_acc_adv, test_acc_x,test_acc_adv])
        # save model
        is_best = test_acc_adv > best_acc
        best_acc = max(test_acc_adv, best_acc)
        save_checkpoint({
                'train':state,
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'test_acc_x': test_acc_x,
                'test_acc_adv':test_acc_adv,
                'train_acc_x':train_acc_x,
                'train_acc_adv':train_acc_adv,
                'train_loss':train_loss,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'schedule':scheduler.state_dict(),
                'global_step':global_step,
            }, is_best, save_path,test_acc_x,test_acc_adv)
    logger.close()
    print('Best acc:',best_acc)

def train(trainloader, criterion, model,ema_model, weight, optimizer, epoch):
    global global_step
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    bar = Bar('training', max=len(trainloader))
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.cuda(), y.cuda()  
       
        loss = criterion(model,ema_model,x, y, optimizer,epoch, args.epsilon, args.step_size, args.num_steps,weight)
        
        # record loss
        losses.update(loss.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= args.start_ema:
            global_step += 1
            update_ema_variables(model,ema_model,args.ema_decay,global_step)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg,global_step

def eval_train(trainloader, model):
    top1_x = AverageMeter()
    top1_adv = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    test_images = 0
    bar = Bar('eval train', max=len(trainloader))
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        x_adv = pgd_linf(model,x,y,args.epsilon,args.epsilon/4,10)
        # measure accuracy and record loss
        prec1_x, _ = accuracy(model(x), y, topk=(1, 5))
        prec1_adv, _ = accuracy(model(x_adv), y, topk=(1, 5))
        top1_x.update(prec1_x.item(), x.size(0))
        top1_adv.update(prec1_adv.item(), x.size(0))
        test_images += x.size(0)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1_x: {top1_x: .4f} | top1_adv: {top1_adv: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1_x=top1_x.avg,
                    top1_adv=top1_adv.avg,
                    )
        bar.next()        
    bar.finish()
    return (top1_x.avg, top1_adv.avg)

def eval_test(testloader, model):
    top1_x = AverageMeter()
    top1_adv = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    bar = Bar('eval test', max=len(testloader))
    for batch_idx, (x, y) in enumerate(testloader):
        x, y = x.cuda(), y.cuda()
        x_adv = pgd_linf(model,x,y,args.epsilon,args.epsilon/4,10)
        # measure accuracy and record loss
        prec1_x, _ = accuracy(model(x), y, topk=(1, 2))
        prec1_adv, _ = accuracy(model(x_adv), y, topk=(1, 2))
        top1_x.update(prec1_x.item(), x.size(0))
        top1_adv.update(prec1_adv.item(), x.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # plot progress
        bar.suffix = '({batch}/{size}) | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1_x: {top1_x: .4f} | top1_adv: {top1_adv: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1_x=top1_x.avg,
                    top1_adv=top1_adv.avg,
                    )
        bar.next()
    bar.finish()
    return (top1_x.avg, top1_adv.avg)

def save_checkpoint(state, is_best, checkpoint, test_n,test_a):
    filepath = os.path.join(checkpoint, 'epoch_{}.pth'.format(str(state['epoch'])))
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth'))


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(alpha).add_(param, alpha=1 - alpha)

def sigmoid_rampup(current, start_es, end_es):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if current < start_es:
        return 0.0
    if current > end_es:
        return 1.0
    else:
        import math
        phase = 1.0 - (current - start_es) / (end_es - start_es)
        return math.exp(-5.0 * phase * phase)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * sigmoid_rampup(epoch, args.start_ema,args.end_ema)
    
if __name__ == '__main__':
    main()

import torch
from attacks import *
from tqdm import tqdm
import os
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from models import ResNet18,WideResNet28
# from main_swa import moving_average,bn_update
parser = argparse.ArgumentParser(description='Adversarial Robustness Test')
parser.add_argument('--workers',type=int,default=0)
parser.add_argument('--dataset',type=str,default='cifar10',choices=['cifar10','cifar100','svhn'])
parser.add_argument('--data_path',type=str,default='../dataset')
parser.add_argument('--model',type=str,default='resnet18',choices=['resnet18','wrn28','wrn34','vgg19'])
parser.add_argument('--model_path',type=str,default='')
parser.add_argument('--test_batch',type=int,default=500)
parser.add_argument('--train_batch',type=int,default=500)
parser.add_argument('--attack',type=str,default='aa',choices=['pgd','aa','cw'])
parser.add_argument('--num-steps', type=list, default=[10,100], help='maximum perturbation step K')

parser.add_argument('--mutigpu',type=bool,default=False)
parser.add_argument('--drop',type=float,default=0)
parser.add_argument('--ema',action="store_true")
parser.add_argument('--log_path',type=str,default='./results.txt')

parser.add_argument('--gpu',type=str,default='0,1')
args = parser.parse_args()
print(args)
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
# load dataset

transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = DataLoader(testset,args.test_batch,shuffle=False,pin_memory=True)
if args.model == 'resnet18':
    model = ResNet18(10)
elif args.model == 'wideresnet28':
    model = WideResNet28(0.0)
model = torch.nn.DataParallel(model).cuda()
# model.cuda()
if args.ema:
    model.load_state_dict(torch.load(args.model_path)['ema_state_dict'])
else:
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
model.eval()

# attack params
epsilon = 8./255
# num_steps = args.num_steps
step_size = 2./255
num_classes = 100 if args.dataset=='cifar100' else 10

correct_n = 0
correct_a = 0
total = 0

with open(args.log_path,'a') as f:
    f.write('model: {}, path:{} \n'.format(args.model,args.model_path))
    if args.attack == 'aa':
        f.write('attack: aa \n')
    elif args.attack == 'pgd':
        f.write('attack: pgd-{} \n'.format(args.num_steps))
# attack
if args.attack == 'aa':
    autoattack_test(model,testloader,10000,batch_size=args.test_batch,version='custom',verbose=True,log_path=args.log_path)
elif args.attack == 'pgd':
    for steps in args.num_steps:
        for i,(x,y) in tqdm(enumerate(testloader)):
            x = x.cuda()
            y = y.cuda()

            x_adv = pgd_linf(model,x,y,epsilon,step_size,steps)
            _, predicted_n = model(x).max(1)
            _, predicted_a = model(x_adv).max(1)

            total += y.size(0)
            correct_n += predicted_n.eq(y).sum().item()
            correct_a += predicted_a.eq(y).sum().item()
        with open(args.log_path,'a') as f:
            f.write("pgd-{}, natural acc: {}, adv acc: {} \n".format(steps,correct_n/total,correct_a/total))
        print("natural acc: {}".format(correct_n/total))
        print("adv acc: {}".format(correct_a/total))
elif args.attack == 'cw':
    for i,(x,y) in tqdm(enumerate(testloader)):
        x = x.cuda()
        y = y.cuda()

        x_adv = cw_linf(model,x,y,num_classes,epsilon,step_size,100)
        # _, predicted_n = model(x).max(1)
        _, predicted_a = model(x_adv).max(1)

        total += y.size(0)
        # correct_n += predicted_n.eq(y).sum().item()
        correct_a += predicted_a.eq(y).sum().item()
    with open(args.log_path,'a') as f:
        f.write("cw-{},  adv acc: {} \n".format(100,correct_a/total))
    print("adv acc: {}".format(correct_a/total))

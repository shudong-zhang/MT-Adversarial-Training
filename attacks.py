import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from autoattack import AutoAttack

def pgd_linf(model,x,y,epsilon,step_size,num_steps,rand_init=True):
    model.eval()
    x_adv = x.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, x.shape)).float().cuda() if rand_init else x.detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(output, y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size* torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv

def autoattack_test(model,testloader,n_ex=10000,batch_size=200,eps=8./255,norm='Linf',version='standard',verbose = True,log_path='./log.txt'):
    '''
    :param model: torch model returns the logits and takes input with components in [0, 1] (NCHW format expected),
    :testloader: testloader,
    :n_ex: total number of images to be attacked,
    :batch_size: you know what batch_size means,
    :param eps: eps is the bound on the norm of the adversarial perturbations,
    :param norm: norm = ['Linf' | 'L2'] is the norm of the threat model,
    :param version:version = ['standard' | 'rand'] 'standard' uses the standard version of AA. 
    :return: attack accuracy
    '''
    # load testloader and prepare dataset to attack
    l_x = []
    l_y = []
    for i, (x, y) in enumerate(testloader):
        l_x.append(x)
        l_y.append(y)
    x_test = torch.cat(l_x, 0)
    y_test = torch.cat(l_y, 0)
    adversary = AutoAttack(model, norm=norm, eps=eps, version=version,log_path=log_path,verbose=verbose)
    with torch.no_grad():
        adv = adversary.run_standard_evaluation(x_test[:n_ex], y_test[:n_ex],bs=batch_size)
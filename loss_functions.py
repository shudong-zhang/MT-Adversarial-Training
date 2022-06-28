import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import random

def AT(model,x,y,optimizer,epsilon=0.031, step_size=0.003,num_steps=10,random_start=True):
    model.eval()
    if random_start:
        x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
    else:
        x_adv = x.detach()
    # x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss


def trades(model,x,y, optimizer,epsilon, step_size, num_steps,beta):
    model.eval()
    
    x_adv = x.detach() + 0.001 * torch.randn_like(x).detach()
    nat_output = model(x)
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(nat_output, dim=1),reduction='sum')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    x_adv = Variable(x_adv, requires_grad=False)
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x)
    adv_logits = model(x_adv)
    
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = F.kl_div(F.log_softmax(adv_logits, dim=1),
                               F.softmax(logits, dim=1),reduction='batchmean')
    loss = loss_natural + beta * loss_robust
    return loss

class mean_teacher():
    def __init__(self,es,consistency_type='mse'):
        self.consistency_type = consistency_type
        self.es = es

    def consistency_loss(self,consistency_weight,logits,ema_logits):
        if self.consistency_type == 'mse':
            return consistency_weight * self.softmax_mse_loss(logits, ema_logits)
        elif self.consistency_type == 'kl':
            return consistency_weight * self.softmax_kl_loss(logits, ema_logits)
        else:
            raise NotImplementedError

    def softmax_mse_loss(self,input_logits, target_logits):
        """Takes softmax on both sides and returns MSE loss

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        batch = input_logits.size(0)
        
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        return F.mse_loss(input_softmax, target_softmax, reduction='sum')/batch

    def softmax_kl_loss(self,input_logits, target_logits):
        """Takes softmax on both sides and returns KL divergence

        Note:
        - Returns the sum over all examples. Divide by the batch size afterwards
          if you want the mean.
        - Sends gradients to inputs but not the targets.
        """
        assert input_logits.size() == target_logits.size()
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        return F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')

    def __call__(self, model,ema_model,x,y, optimizer,epoch,epsilon, step_size, num_steps,consistency_weight,random_start=True):
        model.eval()
        ema_model.train()
        ema_logits = ema_model(x).detach()
        # nat_logits = model(x)
        
        if random_start:
            x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
        else:
            x_adv = x.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                
                if epoch >= self.es:
                    print('haha')    
                    loss = F.cross_entropy(logits, y) + self.consistency_loss(consistency_weight,logits,ema_logits)
                else:
                    loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()
        ema_model.train()
        optimizer.zero_grad()
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        logits = model(x_adv)
        
        if epoch >= self.es:
            ce_loss = F.cross_entropy(logits, y)
            cons_loss = self.consistency_loss(consistency_weight,logits,ema_logits)
            loss = ce_loss+cons_loss
        else:
            ce_loss = F.cross_entropy(logits,y)
            cons_loss = torch.tensor(0)
            loss = ce_loss
        return loss

class trades_teacher(mean_teacher):
    def __init__(self, es, consistency_type,beta):
        super().__init__(es, consistency_type=consistency_type)
        self.beta = beta
    def __call__(self, model, ema_model, x, y, optimizer,epoch, epsilon, step_size, num_steps, consistency_weight):
        model.eval()
        ema_model.train()
        ema_logits = ema_model(x).detach()
        x_adv = x.detach() + 0.001 * torch.randn_like(x).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        nat_logits = model(x)
        for _ in range(num_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_logits = model(x_adv)
                kl_loss = F.kl_div(F.log_softmax(adv_logits, dim=1),
                                        F.softmax(nat_logits, dim=1),reduction='sum')
                if epoch >= self.es:
                    loss = kl_loss + self.consistency_loss(consistency_weight,adv_logits,ema_logits)
                else:
                    loss = kl_loss
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()
        ema_model.train()
        x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        # calculate robust loss
        nat_logits = model(x)
        adv_logits = model(x_adv)
        loss_natural = F.cross_entropy(nat_logits, y)
        loss_robust = F.kl_div(F.log_softmax(adv_logits, dim=1),
                                F.softmax(nat_logits, dim=1),reduction='batchmean')
        trades_loss = loss_natural + self.beta * loss_robust
        if epoch >= self.es:
            cons_loss = self.consistency_loss(consistency_weight,adv_logits,ema_logits)
            loss = trades_loss + cons_loss
        else:
            loss = trades_loss
        return loss

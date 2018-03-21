"""adversary.py"""
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

from models.toynet import ToyNet
from datasets.datasets import return_data
from utils.utils import rm_dir, cuda, where


class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, y, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        h_adv = self.net(x_adv)
        if targeted:
            cost = self.criterion(h_adv, y)
        else:
            cost = -self.criterion(h_adv, y)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)


        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

    def i_fgsm(self, x, y, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        for i in range(iteration):
            h_adv = self.net(x_adv)
            if targeted:
                cost = self.criterion(h_adv, y)
            else:
                cost = -self.criterion(h_adv, y)

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+eps, x+eps, x_adv)
            x_adv = where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

        h = self.net(x)
        h_adv = self.net(x_adv)

        return x_adv, h_adv, h

    def universal(self, args):
        self.set_mode('eval')

        init = False

        correct = 0
        cost = 0
        total = 0

        data_loader = self.data_loader['test']
        for e in range(100000):
            for batch_idx, (images, labels) in enumerate(data_loader):

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))

                if not init:
                    sz = x.size()[1:]
                    r = torch.zeros(sz)
                    r = Variable(cuda(r, self.cuda), requires_grad=True)
                    init = True

                logit = self.net(x+r)
                p_ygx = F.softmax(logit, dim=1)
                H_ygx = (-p_ygx*torch.log(self.eps+p_ygx)).sum(1).mean(0)
                prediction_cost = H_ygx
                #prediction_cost = F.cross_entropy(logit,y)
                #perceptual_cost = -F.l1_loss(x+r,x)
                #perceptual_cost = -F.mse_loss(x+r,x)
                #perceptual_cost = -F.mse_loss(x+r,x) -r.norm()
                perceptual_cost = -F.mse_loss(x+r, x) -F.relu(r.norm()-5)
                #perceptual_cost = -F.relu(r.norm()-5.)
                #if perceptual_cost.data[0] < 10: perceptual_cost.data.fill_(0)
                cost = prediction_cost + perceptual_cost
                #cost = prediction_cost

                self.net.zero_grad()
                if r.grad:
                    r.grad.fill_(0)
                cost.backward()

                #r = r + args.eps*r.grad.sign()
                r = r + r.grad*1e-1
                r = Variable(cuda(r.data, self.cuda), requires_grad=True)



                prediction = logit.max(1)[1]
                correct = torch.eq(prediction, y).float().mean().data[0]
                if batch_idx % 100 == 0:
                    if self.visdom:
                        self.vf.imshow_multi(x.add(r).data)
                        #self.vf.imshow_multi(r.unsqueeze(0).data,factor=4)
                    print(correct*100, prediction_cost.data[0], perceptual_cost.data[0],\
                            r.norm().data[0])

        self.set_mode('train')



    def save_checkpoint(self, filename='ckpt.tar'):
        model_states = {
            'net':self.net.state_dict(),
            }
        optim_states = {
            'optim':self.optim.state_dict(),
            }
        states = {
            'iter':self.global_iter,
            'epoch':self.global_epoch,
            'history':self.history,
            'args':self.args,
            'model_states':model_states,
            'optim_states':optim_states,
            }

        file_path = self.ckpt_dir / filename
        torch.save(states, file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])

            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else: raise('mode error. It should be either train or eval')

    def scale(self, image):
        return image.mul(2).add(-1)

    def unscale(self, image):
        return image.add(1).mul(0.5)

    def summary_flush(self, silent=True):
        rm_dir(self.summary_dir, silent)

    def checkpoint_flush(self, silent=True):
        rm_dir(self.ckpt_dir, silent)

import numpy as np
import os, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

from models.toynet import TOYNET
from datasets.datasets import return_data
from utils.utils import One_Hot, rm_dir, cuda


class Solver(object):
    def __init__(self, args):
        self.args = args

        # Basic
        self.cuda = (args.cuda and torch.cuda.is_available())
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eps = args.eps
        self.lr = args.lr
        self.y_dim = args.y_dim
        self.onehot_module = One_Hot(args.y_dim)
        self.dataset = args.dataset
        self.data_loader = return_data(args)
        self.global_epoch = 0
        self.global_iter = 0
        self.print_ = not args.silent

        self.env_name = args.env_name
        self.tensorboard = args.tensorboard
        if self.tensorboard : from tensorboardX import SummaryWriter
        self.visdom = args.visdom
        if self.visdom : from utils.visdom_utils import VisFunc

        self.ckpt_dir = Path(args.ckpt_dir).joinpath(args.env_name)
        if not self.ckpt_dir.exists() : self.ckpt_dir.mkdir(parents=True,exist_ok=True)
        self.output_dir = Path(args.output_dir).joinpath(args.env_name)
        if not self.output_dir.exists() : self.output_dir.mkdir(parents=True,exist_ok=True)

        # Visualization Tools
        self.visualization_init(args)

        # Histories
        self.history = dict()
        self.history['acc']=0.
        self.history['epoch']=0
        self.history['iter']=0

        # Models & Optimizers
        self.model_init(args)
        self.load_ckpt = args.load_ckpt
        if self.load_ckpt != '' : self.load_checkpoint(self.load_ckpt)

        # Losses & Distances
        self.CE_Loss = cuda(nn.CrossEntropyLoss(),self.cuda)


    def visualization_init(self, args):
        # Visdom
        if self.visdom :
            self.port = args.visdom_port
            self.vf = VisFunc(enval=self.env_name,port=self.port)

        # TensorboardX
        if self.tensorboard :
            self.summary_dir = Path(args.summary_dir).joinpath(args.env_name)
            if not self.summary_dir.exists() : self.summary_dir.mkdir(parents=True,exist_ok=True)

            self.tf = SummaryWriter(log_dir=str(self.summary_dir))
            self.tf.add_text(tag='argument',text_string=str(args),global_step=self.global_epoch)

    def model_init(self, args):
        # Network
        self.net = cuda(TOYNET(y_dim=self.y_dim),self.cuda)
        self.net.weight_init(type='kaiming')

        # Optimizers
        self.optim = optim.Adam([{'params':self.net.parameters(),'lr':self.lr}],
                                    betas=(0.5, 0.999))

    def train(self):
        self.set_mode('train')
        for e in range(self.epoch) :
            self.global_epoch += 1

            correct = 0.
            cost = 0.
            total = 0.
            for batch_idx, (images,labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = cuda(Variable(images),self.cuda)
                y = cuda(Variable(labels),self.cuda)

                logit = self.net(x)
                prediction = logit.max(1)[1]

                correct = torch.eq(prediction,y).float().mean().data[0]
                cost = F.cross_entropy(logit,y)

                self.optim.zero_grad()
                cost.backward()
                self.optim.step()

                if batch_idx % 100 == 0 :
                    if self.print_: print()
                    if self.print_: print(self.env_name)
                    if self.print_: print('[{:03d}:{:03d}]'.format(self.global_epoch,batch_idx))
                    if self.print_: print('acc:{:.3f} loss:{:.3f}'.format(correct,cost.data[0]))


                    if self.tensorboard :
                        self.tf.add_scalars(main_tag='performance/acc',
                                            tag_scalar_dict={'train':correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/error',
                                            tag_scalar_dict={'train':1-correct},
                                            global_step=self.global_iter)
                        self.tf.add_scalars(main_tag='performance/cost',
                                            tag_scalar_dict={'train':cost.data[0]},
                                            global_step=self.global_iter)


            self.test()


        if self.tensorboard :
            self.tf.add_scalars(main_tag='performance/best/acc',
                                tag_scalar_dict={'test':self.history['acc']},
                                global_step=self.history['iter'])
        print(" [*] Training Finished!")



    def test(self):
        self.set_mode('eval')

        correct = 0.
        cost = 0.
        total = 0.

        data_loader = self.data_loader['test']
        for batch_idx, (images,labels) in enumerate(data_loader):
            x = cuda(Variable(images),self.cuda)
            y = cuda(Variable(labels),self.cuda)

            logit = self.net(x)
            prediction = logit.max(1)[1]

            correct += torch.eq(prediction,y).float().sum().data[0]
            cost += F.cross_entropy(logit,y,size_average=False).data[0]
            total += x.size(0)

        accuracy = correct / total
        cost /= total


        if self.print_:
            print()
            print('[{:03d}]\nTEST RESULT'.format(self.global_epoch))
            print('ACC:{:.4f}'.format(accuracy))
            print('*TOP* ACC:{:.4f} at e:{:03d}'.format(accuracy,self.global_epoch,))
            print()

            if self.tensorboard :
                self.tf.add_scalars(main_tag='performance/acc',
                                    tag_scalar_dict={'test':accuracy},
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/error',
                                    tag_scalar_dict={'test':(1-accuracy)},
                                    global_step=self.global_iter)

                self.tf.add_scalars(main_tag='performance/cost',
                                    tag_scalar_dict={'test':cost},
                                    global_step=self.global_iter)

        if self.history['acc'] < accuracy :
            self.history['acc'] = accuracy
            self.history['epoch'] = self.global_epoch
            self.history['iter'] = self.global_iter
            self.save_checkpoint('best_acc.tar')

        self.set_mode('train')


    def generate(self,num_sample=100,epsilon=0.03,iteration=10):
        self.set_mode('eval')

        x_true, y_true = self.sample_data(num_sample)
        x_adv, changed, values = self.FGSM(x_true,y_true,epsilon,iteration)
        accuracy_true,cost_true,accuracy_adv,cost_adv = values

        save_image(x_true,
                self.output_dir.joinpath('legitimate(e:{},i:{}).jpg'.format(epsilon,iteration)),
                nrow=10,
                padding=0)
        save_image(x_adv,
                self.output_dir.joinpath('perturbed(e:{},i:{}).jpg'.format(epsilon,iteration)),
                nrow=10,
                padding=0)
        save_image(changed,
                self.output_dir.joinpath('changed(e:{},i:{}).jpg'.format(epsilon,iteration)),
                nrow=10,
                padding=0)
        if self.visdom :
            self.vf.imshow_multi(x_true, title='legitimate', factor=1.5)
            self.vf.imshow_multi(x_adv, title='perturbed(e:{},i:{})'.format(epsilon,iteration), factor=1.5)
            self.vf.imshow_multi(changed, title='changed(white)'.format(epsilon), factor=1.5)

        print('[BEFORE] accuracy : {:.2f} cost : {:.3f}'.format(accuracy_true,cost_true))
        print('[AFTER] accuracy : {:.2f} cost : {:.3f}'.format(accuracy_adv,cost_adv))

        self.set_mode('train')


    def sample_data(self, num_sample=100):

        total = len(self.data_loader['test'].dataset)
        seed = torch.FloatTensor(num_sample).uniform_(1,total).long()

        x = self.data_loader['test'].dataset.test_data[seed]
        x = self.scale(x.float().unsqueeze(1).div(255))
        y = self.data_loader['test'].dataset.test_labels[seed]

        return x,y


    def FGSM(self, x, y, epsilon=0.03, iteration=1):
        self.set_mode('eval')

        y_true = cuda(Variable(y, requires_grad=False),self.cuda)
        x_true = cuda(Variable(x, requires_grad=True),self.cuda)
        x_adv = x_true

        logit_true = self.net(x_true)
        prediction_true = logit_true.max(1)[1]
        accuracy_true = torch.eq(prediction_true,y_true).float().mean()
        cost_true = F.cross_entropy(logit_true,y_true)

        for i in range(iteration):
            logit = self.net(x_adv)
            cost = F.cross_entropy(logit, y_true)
            self.net.zero_grad()
            try :x_adv.grad.data.fill_(0)
            except : pass
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv + epsilon*x_adv.grad
            x_adv = torch.clamp(x_adv,-1,1)
            x_adv = Variable(x_adv.data, requires_grad=True)

        logit_adv = self.net(x_adv)
        prediction_adv = logit_adv.max(1)[1]
        accuracy_adv = torch.eq(prediction_adv,y_true).float().mean()
        cost_adv = F.cross_entropy(logit_adv, y_true)

        # make indication of perturbed images that changed predictions of the classifier
        changed = torch.eq(prediction_true,prediction_adv)
        changed = torch.eq(changed,0)
        changed = changed.view(-1,1,1,1).repeat(1,1,28,28)

        temp = changed.float()
        temp = temp.repeat(1,3,1,1)

        r = temp[:,0,:,:].clone()
        r *= 252
        g = temp[:,1,:,:].clone()
        g *= 207
        b = temp[:,2,:,:].clone()
        b *= 25

        changed = torch.stack([r,g,b],1)
        changed = self.scale(changed/255)
        changed[:,:,2:-1,2:-1] = x_adv.repeat(1,3,1,1)[:,:,2:-1,2:-1]

        self.set_mode('train')

        return x_adv.data, changed.data,\
                (accuracy_true.data[0],cost_true.data[0],accuracy_adv.data[0],cost_adv.data[0])




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
        torch.save(states,file_path.open('wb+'))
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))


    def load_checkpoint(self, filename='best_acc.tar'):
        file_path = self.ckpt_dir / filename
        if file_path.is_file():
            print("=> loading checkpoint '{}'".format(file_path,self.global_iter))
            checkpoint = torch.load(file_path.open('rb'))
            self.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']
            self.history = checkpoint['history']

            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])

            print("=> loaded checkpoint '{} (iter {})'".format(
                file_path, self.global_iter))

        else:
            print("=> no checkpoint found at '{}'".format(file_path))


    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.net.train()
        elif mode == 'eval' :
            self.net.eval()
        else : raise('mode error. It should be either train or eval')


    def scale(self, image):
        return image.mul(2).add(-1)


    def unscale(self, image):
        return image.add(1).mul(0.5)


    def summary_flush(self, silent=True):
        rm_dir(self.summary_dir, silent)


    def checkpoint_flush(self, silent=True):
        rm_dir(self.ckpt_dir, silent)



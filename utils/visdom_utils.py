import visdom
from scipy.misc import imresize
import numpy as np
from torchvision.utils import make_grid

class VisFunc(object):

    def __init__(self, config=None, vis=None, enval='hproto',port=8097):
        self.config = config
        self.vis = visdom.Visdom(env=enval, port=port)
        self.win = None
        self.win2 = None
        self.epoch_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.epoch_list2 = []
        self.train_acc_list = []
        self.val_acc_list = []


    def imshow(self, img, title=' ', caption=' ', factor=1):

        img = img / 2 + 0.5      # Unnormalize
        npimg = img.numpy()
        obj = np.transpose(npimg, (1,2,0))
        obj = np.swapaxes(obj,0,2)
        obj = np.swapaxes(obj,1,2)

        imgsize = tuple((np.array(obj.shape[1:])*factor).astype(int))
        rgbArray = np.zeros(tuple([3])+imgsize,'float32')
        rgbArray[0,...] = imresize(obj[0,:,:],imgsize,'cubic')
        rgbArray[1,...] = imresize(obj[1,:,:],imgsize,'cubic')
        rgbArray[2,...] = imresize(obj[2,:,:],imgsize,'cubic')

        self.vis.image( rgbArray,
                        opts=dict(title=title, caption=caption),
                       )


    def imshow_multi(self, imgs, nrow=10, title=' ', caption=' ', factor=1):
        #self.imshow( make_grid(imgs,nrow,padding=padding), title, caption, factor)
        self.imshow( make_grid(imgs,nrow), title, caption, factor)


    def imshow_one_batch(self, loader, classes=None, factor=1):
        dataiter = iter(loader)
        images, labels = dataiter.next()
        self.imshow(make_grid(images,padding))

        if classes:
            print(' '.join('%5s' % classes[labels[j]]
                                    for j in range(loader.batch_size)))
        else:
            print(' '.join('%5s' % labels[j]
                                    for j in range(loader.batch_size)))


    def plot(self, epoch, train_loss, val_loss,Des):
        ''' plot learning curve interactively with visdom '''
        self.epoch_list.append(epoch)
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)

        if not self.win:
            # send line plot
            # embed()
            self.win = self.vis.line(
                X=np.array(self.epoch_list),
                Y=np.array([[self.train_loss_list[-1], self.val_loss_list[-1]]]),
                opts=dict(
                    title='Learning Curve (' + Des +')',
                    xlabel='Epoch',
                    ylabel='Loss',
                    legend=['train_loss', 'val_loss'],
                    #caption=Des
                ))
            # send text memo (configuration)
           #  self.vis.text(str(Des))
        else:
            self.vis.updateTrace(
                X=np.array(self.epoch_list[-2:]),
                Y=np.array(self.train_loss_list[-2:]),
                win=self.win,
                name='train_loss',
            )
            self.vis.updateTrace(
                X=np.array(self.epoch_list[-2:]),
                Y=np.array(self.val_loss_list[-2:]),
                win=self.win,
                name='val_loss',
            )


    def acc_plot(self, epoch, train_acc, val_acc, Des):
        ''' plot learning curve interactively with visdom '''
        self.epoch_list2.append(epoch)
        self.train_acc_list.append(train_acc)
        self.val_acc_list.append(val_acc)

        if not self.win2:
            # send line plot
            # embed()
            self.win2 = self.vis.line(
                X=np.array(self.epoch_list2),
                Y=np.array([[self.train_acc_list[-1], self.val_acc_list[-1]]]),
                opts=dict(
                    title='Accuracy Curve (' + Des +')',
                    xlabel='Epoch',
                    ylabel='Accuracy',
                    legend=['train_accuracy', 'val_accuracy']
                ))
            # send text memo (configuration)
            # self.vis.text(str(self.config))
        else:
            self.vis.updateTrace(
                X=np.array(self.epoch_list2[-2:]),
                Y=np.array(self.train_acc_list[-2:]),
                win=self.win2,
                name='train_accuracy',
            )
            self.vis.updateTrace(
                X=np.array(self.epoch_list2[-2:]),
                Y=np.array(self.val_acc_list[-2:]),
                win=self.win2,
                name='val_accuracy',
            )


    def plot2(self, epoch, train_loss, val_loss,Des, win):
        ''' plot learning curve interactively with visdom '''
        self.epoch_list.append(epoch)
        self.train_loss_list.append(train_loss)
        self.val_loss_list.append(val_loss)

        if not self.win:
            self.win = win
            # send line plot
            # embed()
            #self.win = self.vis.line(
            #    X=np.array(self.epoch_list),
            #    Y=np.array([[self.train_loss_list[-1], self.val_loss_list[-1]]]),
            #    opts=dict(
            #        title='Learning Curve (' + Des +')',
            #        xlabel='Epoch',
            #        ylabel='Loss',
            #        legend=['train_loss', 'val_loss'],
            #        #caption=Des
            #    ))
            ## send text memo (configuration)
           #  self.vis.text(str(Des))
        else:
            self.vis.updateTrace(
                X=np.array(self.epoch_list[-2:]),
                Y=np.array(self.train_loss_list[-2:]),
                win=self.win,
                name='train_loss2',
            )
            self.vis.updateTrace(
                X=np.array(self.epoch_list[-2:]),
                Y=np.array(self.val_loss_list[-2:]),
                win=self.win,
                name='val_lossi2',
            )

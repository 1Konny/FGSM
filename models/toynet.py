import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TOYNET(nn.Module):
    def __init__(self, x_dim=784, y_dim=10, p=0.2):
        super(TOYNET, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.x_dim,300),
            nn.ReLU(True),
            nn.Linear(300,150),
            nn.ReLU(True),
            nn.Linear(150,self.y_dim)
            )

    def forward(self, X):
        if X.dim() > 2 : X = X.view(X.size(0),-1)
        out = self.mlp(X)

        return out

    def weight_init(self,type='kaiming'):
        if type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())

def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            try:m.bias.data.zero_()
            except:pass
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            try:m.bias.data.zero_()
            except:pass

def kaiming_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform(m.weight,a=0,mode='fan_in')
            try:m.bias.data.zero_()
            except:pass
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            try:m.bias.data.zero_()
            except:pass

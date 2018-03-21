"""toynet.py"""
import torch.nn as nn

class ToyNet(nn.Module):
    def __init__(self, x_dim=784, y_dim=10):
        super(ToyNet, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.x_dim, 300),
            nn.ReLU(True),
            nn.Linear(300, 150),
            nn.ReLU(True),
            nn.Linear(150, self.y_dim)
            )

    def forward(self, X):
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        out = self.mlp(X)

        return out

    def weight_init(self, _type='kaiming'):
        if _type == 'kaiming':
            for ms in self._modules:
                kaiming_init(self._modules[ms].parameters())


def xavier_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()


def kaiming_init(ms):
    for m in ms:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform(m.weight, a=0, mode='fan_in')
            if m.bias.data:
                m.bias.data.zero_()
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.weight.data.fill_(1)
            if m.bias.data:
                m.bias.data.zero_()

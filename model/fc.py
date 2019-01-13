import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

'''
Code from https://github.com/hengyuan-hu/bottom-up-attention-vqa/blob/master/fc.py
'''
class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims, norm=True):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            if norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))

            layers.append(nn.ReLU())
        if norm:
            layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        else:
            layers.append(nn.Linear(dims[-2], dims[-1]))
        
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

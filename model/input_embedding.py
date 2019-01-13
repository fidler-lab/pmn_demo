import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
import sys, os
sys.path.insert(0, '..')


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, vec_dim=300, glove_wemb=None):
        super(InputEmbedding, self).__init__()
        self.name = 'InputEmbedding'       
        self.vocab_size = vocab_size
        self.vec_dim = vec_dim
        self.wembed = nn.Embedding(self.vocab_size+1, self.vec_dim)
        if glove_wemb is not None:
            self.wembed.weight.data.copy_(torch.from_numpy(glove_wemb))
   
    def forward(self, inds):
        inds = inds.type(torch.LongTensor).cuda(0)
        wvec = self.wembed(inds)
        return wvec

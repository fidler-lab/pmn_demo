import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
import sys, os
sys.path.insert(0, '..')
from utils import *


cuda_available = torch.cuda.is_available()

class TaskModuleTag(nn.Module):
    def __init__(self, c_input_dim, task_v_size, cnn_feature_dim, map_dim, tag_num_class, att_num_class, num_glimpse, tag_wemb=None):
        super(TaskModuleTag, self).__init__()
        self.name = 'TaskModuleTag'    
        self.c_input_dim = c_input_dim   
        self.cnn_feature_dim = cnn_feature_dim
        #self.reduced_dim = cnn_feature_dim/4
        self.reduced_dim = 300
        self.map_dim = map_dim
        self.tag_num_class = tag_num_class
        self.att_num_class = att_num_class
        self.num_glimpse = num_glimpse
        self.task_v_size = task_v_size
        self.wembed = nn.Embedding(self.tag_num_class, 300)
        if tag_wemb is not None:
            self.wembed.weight.data.copy_(torch.from_numpy(tag_wemb))
        self.reduce_input_dim = nn.Sequential(nn.Linear(300, self.c_input_dim),
                                        nn.Tanh())
        self.reduce_obj = nn.Linear(cnn_feature_dim, self.reduced_dim)
        self.tag_output = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.reduced_dim, tag_num_class),
        )
        if tag_wemb is not None:
            self.tag_output[1].weight.data.copy_(torch.from_numpy(tag_wemb))
            self.tag_output[1].weight.requires_grad=False
            self.tag_output[1].bias.data.copy_(torch.from_numpy(np.array([0.0]*(tag_num_class))))
            self.tag_output[1].bias.requires_grad=False
    
    def forward(self, input, maps=None, have_maps=True, emb_dim=300, mode='reduce_img'):
        '''
        '''
        if mode == 'reduce_img':
            obj = self.reduce_obj(input)
            obj = F.tanh(obj)
            if have_maps:
                maps = maps.view(-1, 1, 14, 14)
                obj = (obj*maps).view(-1, self.reduced_dim, self.map_dim)
                obj = torch.sum(obj, dim=2)
            output = obj
             
        elif mode == 'embedding':
            inds = input.type(torch.LongTensor)
            if cuda_available:
                inds = inds.cuda(0)
            wvec = self.wembed(inds).detach()
            if emb_dim != 300:
                wvec = self.reduce_input_dim(wvec)
            output = wvec
        elif mode == 'obj':
            output = self.tag_output(input)
        else:
            print 'wrong mode in task_module_tag'
            exit(-1)
        return output 

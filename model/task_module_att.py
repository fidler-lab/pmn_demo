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
from model.fc import FCNet
from model.attention import ImgAttention

cuda_available = torch.cuda.is_available()

class TaskModuleAtt(nn.Module):
    def __init__(self, c_input_dim, task_v_size, cnn_feature_dim, map_dim, tag_num_class, att_num_class, num_glimpse, att_wemb=None):
        super(TaskModuleAtt, self).__init__()
        self.name = 'TaskModuleAtt'    
        self.c_input_dim = c_input_dim   
        self.cnn_feature_dim = cnn_feature_dim
        #self.reduced_dim = cnn_feature_dim/4
        self.reduced_dim = 300
        self.map_dim = map_dim
        self.tag_num_class = tag_num_class
        self.att_num_class = att_num_class
        self.num_glimpse = num_glimpse
        self.task_v_size = task_v_size
        self.wembed = nn.Embedding(self.att_num_class, 300)
        if att_wemb is not None:
            self.wembed.weight.data.copy_(torch.from_numpy(att_wemb))
        self.reduce_input_dim = nn.Sequential(nn.Linear(300, self.c_input_dim),
                                        nn.Tanh())
        self.reduce_att = nn.Sequential(nn.Linear(cnn_feature_dim, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, self.reduced_dim))

        #self.reduce_att = nn.Linear(cnn_feature_dim, self.reduced_dim)


        self.att_output = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.reduced_dim, att_num_class),
        )
        if att_wemb is not None:
            self.att_output[1].weight.data.copy_(torch.from_numpy(att_wemb))
            self.att_output[1].weight.requires_grad=False
            self.att_output[1].bias.data.copy_(torch.from_numpy(np.array([0.0]*(att_num_class))))
            self.att_output[1].bias.requires_grad=False
    
    def forward(self, input, maps=None, have_maps=True, emb_dim=300, mode='reduce_img'):
        '''
        '''
        if mode == 'reduce_img':
            att = self.reduce_att(input)
            att = F.tanh(att)
            if have_maps:
                maps = maps.view(-1, 1, 14, 14)
                att = (att*maps).view(-1, self.reduced_dim, self.map_dim)
                att = torch.sum(att, dim=2)
            output =  att
             
        elif mode == 'embedding':
            inds = input.type(torch.LongTensor)
            if cuda_available:
                inds = inds.cuda(0)
            wvec = self.wembed(inds).detach()
            if emb_dim != 300:
                wvec = self.reduce_input_dim(wvec)
            output = wvec
        elif mode == 'att':
            output = self.att_output(input)
        else:
            print 'wrong mode in task_module_tag'
            exit(-1)
        return output 

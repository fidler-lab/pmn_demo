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

class TaskModuleRel(nn.Module):
    def __init__(self, obj_dim, att_dim, K, num_rel_class, rel_vec_dim, num_glimpse):
        super(TaskModuleRel, self).__init__()
        self.name = 'TaskModuleRel'    
        self.task_v_size = obj_dim
        self.obj_dim = obj_dim
        self.att_dim = att_dim
        self.K = K
        self.num_rel_class = num_rel_class
        self.num_glimpse = num_glimpse
        self.vec_size = rel_vec_dim
        self.box_proj = nn.Sequential(nn.Linear(5, self.vec_size/2),
                                      nn.ReLU(),
                                      nn.Linear(self.vec_size/2, self.vec_size/2))
        self.X_proj = nn.Sequential(nn.Linear(self.obj_dim, self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, self.vec_size/2))
        self.f_proj = nn.Sequential(nn.ReLU(),
                                    nn.Linear(self.vec_size, self.vec_size),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size, self.vec_size),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size, self.vec_size))
        self.j_proj_obj = nn.Linear(self.vec_size, self.vec_size)
        self.j_proj_sub = nn.Linear(self.vec_size, self.vec_size)
        self.linear_obj = nn.Sequential(nn.Linear(self.vec_size,self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, 1))
        self.linear_sub = nn.Sequential(nn.Linear(self.vec_size,self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, self.vec_size/2),
                                    nn.ReLU(),
                                    nn.Linear(self.vec_size/2, 1))
        self.rel_embed = nn.Linear(self.num_rel_class, self.vec_size, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.dropout_v = nn.Dropout(0.2)
        self.dropout_r = nn.Dropout(0.2)
        

    def logits(self, X, j, input_kind='obj'):
        batch, k, _ = X.size()
        X_proj = X 
        if input_kind == 'obj':
            j_proj = self.j_proj_obj(j).unsqueeze(1).repeat(1, k, 1)
        else:
            j_proj = self.j_proj_sub(j).unsqueeze(1).repeat(1, k, 1)
        joint_repr = X_proj * j_proj
        joint_repr = self.dropout(joint_repr)
        if input_kind == 'obj':
            logits = self.linear_obj(joint_repr)
        else:
            logits = self.linear_sub(joint_repr)
      
        return logits
    
    def forward(self, obj_vec=None, boxes=None, rel_obj=None, rel_rel=None, rel_sub=None, X_loc=None, f=None, v_rel_obj=None, 
        	v_rel_sub=None, get_obj=False, get_sub=False, mode='', step=0):
        

        if mode == 'get_rel_emb':
            tmp = Variable(torch.FloatTensor(np.eye(self.num_rel_class)).cuda(0), requires_grad=False)
            emb = self.rel_embed(tmp)
            return emb
        if mode =='get_f':
            box_repr = self.box_proj(boxes)
            X_repr = self.X_proj(X_loc)
            f = torch.cat([box_repr, X_repr], dim=2)
            f = self.f_proj(f)
            return f
        if mode == 'joint':
            if get_obj:
                rel_obj = rel_obj.view(-1, self.K, 1)
                v_obj = f * rel_obj
                v_obj = torch.sum(v_obj, dim=1)
            if get_sub: 
                rel_sub = rel_sub.view(-1, self.K, 1)
                v_sub = f * rel_sub
                v_sub = torch.sum(v_sub, dim=1)
            
            r = self.rel_embed(rel_rel)
            r = self.dropout_r(r)
            ret = []
            if get_obj:
                v_obj = self.dropout_v(v_obj)
                joint_obj = v_obj * r
                ret.append(joint_obj)
            if get_sub:
                v_sub = self.dropout_v(v_sub)
                joint_sub = v_sub * r
                ret.append(joint_sub)
            return ret
        elif mode == 'rel':
            logits_obj = self.logits(f, v_rel_obj, input_kind='obj')
            logits_sub = self.logits(f, v_rel_sub, input_kind='sub')
            return logits_obj, logits_sub
        else:
            print 'wrong mode in task_module_rel'
            exit(-1)
        return output 



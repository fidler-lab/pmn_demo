import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from attention import ImgAttention
cuda_available = torch.cuda.is_available()

class RELController(nn.Module):
    def __init__(self, task_v_size, att_dim, num_rel_classes):
        super(RELController, self).__init__()
        self.name = 'RELController' 
        self.task_v_size = task_v_size
        self.att_dim = att_dim
        self.num_rel_classes = num_rel_classes

        
    def forward(self, X_rel=None, rel_obj=None, rel_rel=None, rel_sub=None, wvecs=None, q_vec=None, ml=None, X_loc=None, mode='', step=0):
        if mode == 'execute':
            if rel_sub is None:
                v_rel_input = ml['TaskModuleRel'](f=X_rel, rel_obj=rel_obj, rel_rel=rel_rel, get_obj=True, mode='joint')
                y_rel_obj, y_rel_sub = ml['TaskModuleRel'](f=X_rel, v_rel_obj=v_rel_input[0], v_rel_sub=v_rel_input[0], mode='rel')
            else:
                v_rels = ml['TaskModuleRel'](f=X_rel, rel_obj=rel_obj, rel_rel=rel_rel, rel_sub=rel_sub, get_obj=True, get_sub=True, mode='joint', step=step)
                v_rel_obj, v_rel_sub = v_rels
                y_rel_obj, y_rel_sub = ml['TaskModuleRel'](f=X_rel,  v_rel_obj=v_rel_obj, v_rel_sub=v_rel_sub, mode='rel')
            rel_output = [y_rel_obj, y_rel_sub]
            return rel_output









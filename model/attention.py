import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from model.fc import FCNet
from torch.nn.utils.weight_norm import weight_norm

class ImgAttention(nn.Module):
    def __init__(self, q_dim, v_dim, num_hid, num_boxes, att_mode='softmax', norm=True, do_dropout=True):
        super(ImgAttention, self).__init__()
        self.name = 'ImgAttention'
        self.v_proj = FCNet([v_dim, num_hid], norm=norm)
        self.q_proj = FCNet([q_dim, num_hid], norm=norm)
        self.do_dropout = do_dropout
        self.dropout = nn.Dropout(0.5)
        self.gt_W = nn.Linear(v_dim + q_dim, num_hid)
        self.gt_W_prime = nn.Linear(v_dim + q_dim, num_hid)
        
        self.att_mode = att_mode
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)
        self.K = num_boxes

    
    def _gated_tanh(self, x, W, W_prime):
        y_tilde = F.tanh(W(x))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y, g

    def forward(self, v, q, out=False):
        logits, joint_repr = self.logits(v, q, out=out)
        logits = logits.view(-1, self.K)

        if self.att_mode == 'sigmoid':
            sig_maps = nn.functional.sigmoid(logits)
        else:
            sig_maps = nn.functional.softmax(logits)
        maps = sig_maps
        maps = maps.view(-1, self.K, 1)
        f = (v * maps).sum(1)
        return f, maps, sig_maps, logits, joint_repr

    def logits(self, v, q, out=False):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v)
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        if not hasattr(self, 'do_dropout'):
            joint_repr = self.dropout(joint_repr)
        else:
            if self.do_dropout:
                joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits, joint_repr

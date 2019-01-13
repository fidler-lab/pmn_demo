import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.insert(0, '..')
from utils import *

cuda_available = torch.cuda.is_available()

class TaskModuleVqa(nn.Module):
    def __init__(self, feat_dim, obj_dim, att_dim, hid_dim, task_v_size, c_num_hidden, img_map_size, num_class, num_glimpse, 
                c_input_size, q_vocab_size, q_len, q_emb_size):
        super(TaskModuleVqa, self).__init__()
        self.name = 'TaskModuleVqa'       
        self.feat_dim = feat_dim
        self.obj_dim = obj_dim
        self.att_dim = att_dim
        self.hid_dim = hid_dim
        self.img_map_size = img_map_size
        self.num_class = num_class
        self.num_glimpse = num_glimpse
        self.c_input_size = c_input_size
        self.c_num_hidden = c_num_hidden
        self.q_vocab_size = q_vocab_size
        self.q_len = q_len
        self.q_emb_size = q_emb_size
        self.num_class = num_class
        self.task_v_size = task_v_size

        # Linear 
        self.gt_W_question = nn.Linear(task_v_size, hid_dim)
        self.gt_W_prime_question = nn.Linear(task_v_size, hid_dim)
        self.gt_W_img = nn.Linear(feat_dim, hid_dim)
        self.gt_W_prime_img = nn.Linear(feat_dim, hid_dim)
        self.K = img_map_size*img_map_size
        self.h_W_img = nn.Linear(hid_dim, feat_dim)
        self.h_W_prime_img = nn.Linear(hid_dim, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_class)
        self.gate_out = nn.Sequential(nn.Linear(hid_dim+feat_dim, 1),
                                    nn.Sigmoid())

        self.gt_W_knowledge = nn.Linear(task_v_size, hid_dim)
        self.gt_W_prime_knowledge = nn.Linear(task_v_size, hid_dim)
        self.h_W_n = nn.Linear(hid_dim, feat_dim)
        self.h_W_prime_n = nn.Linear(hid_dim, feat_dim)



    def forward(self, k_ts=[], Q=None, v_head=None, mode='step'):
        if mode == 'v_vqa':
            v_vqa, v_vqa_g = self._gated_tanh(v_head, self.gt_W_img, self.gt_W_prime_img, None)
            return v_vqa
        elif mode == 'output':
            y_vqa, vqa_vecs = 0, []
            Qgt, Qgt_prev_g = self._gated_tanh(Q, self.gt_W_question, self.gt_W_prime_question, None)
            for k_t in k_ts:
                k_t, k_t_g = self._gated_tanh(k_t, self.gt_W_knowledge, self.gt_W_prime_knowledge, None)
                h = torch.mul(Qgt, k_t)
                n_h, n_h_g = self._gated_tanh(h, self.h_W_n, self.h_W_prime_n, None)
                y_vqa += self.classifier(n_h)
            return y_vqa, vqa_vecs 
            

    def _gated_tanh(self, x, W, W_prime, bn):
        """
        Implement the gated hyperbolic tangent non-linear activation
            x: input tensor of dimension m
        """
        if bn == None:
            y_tilde = F.tanh(W(x))
        else:
            y_tilde = F.tanh(bn(W(x)))
        g = F.sigmoid(W_prime(x))
        y = torch.mul(y_tilde, g)
        return y, g

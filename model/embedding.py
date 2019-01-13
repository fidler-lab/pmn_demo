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

class QuestionEmbedding(nn.Module):
    def __init__(self, c_input_size, q_vocab_size, q_len, q_emb_size, vqa_wemb=None):
        super(QuestionEmbedding, self).__init__()
        self.name = 'QuestionEmbedding'
        self.c_input_size = c_input_size
        self.q_vocab_size = q_vocab_size
        self.q_len = q_len
        self.q_emb_size = q_emb_size
       
        #self.q_rnn = nn.GRU(q_emb_size, c_input_size, 1, dropout=0.5)
        self.q_rnn = nn.GRU(q_emb_size, c_input_size/2, 1, dropout=0.0, bidirectional=True)
        self.wembed = nn.Embedding(self.q_vocab_size+3, self.q_emb_size)
        if vqa_wemb is not None:
            self.wembed.weight.data.copy_(torch.from_numpy(vqa_wemb))
        self.input_reg = nn.Sequential(nn.ReLU(), nn.Dropout())

    def forward(self, k_ts=[], Q=None, tmp=None, reg=False, v_head=None, q_inds=None, mode='embedding'):
        '''
        run question vector representation if mode == 'question_vec'
        run rnn step if mode == 'step'
        '''
        inds = q_inds.type(torch.LongTensor)
        if cuda_available:
            inds = inds.cuda(0)
        emb = self.wembed(inds)
        #emb = self.input_reg(emb)
        
        inds = emb.permute(1,0,2)
        enc, hid = self.q_rnn(inds)
        first = enc[0]
        last = enc[-1]
        first = torch.split(first, self.c_input_size/2, dim=1)[1]
        last = torch.split(last, self.c_input_size/2, dim=1)[0]
        qvec = torch.cat([first, last], dim=1)
        
        
        #qvec = enc[-1]
        
        return qvec, enc, emb

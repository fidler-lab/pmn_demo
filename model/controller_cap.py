import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from attention import ImgAttention
cuda_available = torch.cuda.is_available()

class CAPController(nn.Module):
    def __init__(self, K, cap_hidden_dim, cap_emb_dim, vocab_size, cap_att_dim, X_loc_dim, num_tasks, cnt_num_class, vqa_num_class):
        super(CAPController, self).__init__()
        self.name = 'CAPController' 
        self.K = K
        self.cap_hidden_dim = cap_hidden_dim
        self.cap_emb_dim = cap_emb_dim
        self.vocab_size = vocab_size
        self.cap_att_dim = cap_att_dim
        self.X_loc_dim = X_loc_dim
        self.num_tasks = num_tasks
        self.cnt_num_class = cnt_num_class
        self.vqa_num_class = vqa_num_class

        task_v_size = 1000

        self.gate_objsub = nn.Sequential(nn.Linear(self.cap_hidden_dim, 2),)
        self.gate_rel = nn.Sequential(nn.Linear(self.cap_hidden_dim, 2),)
        self.gate_hv = nn.Sequential(nn.Linear(self.cap_hidden_dim, 2),
                                     nn.Softmax())
        self.rel_rel_attention = ImgAttention(cap_hidden_dim, 512, cap_att_dim, 21, att_mode='softmax')
        self.cap_img_attention = ImgAttention(cap_hidden_dim, X_loc_dim, cap_att_dim, self.K, att_mode='softmax')
        self.i_n_gate = nn.Sequential(nn.Linear(self.cap_hidden_dim, num_tasks),
                                      nn.Softmax())    
        self.vqa_input = nn.Sequential(nn.Linear(self.cap_hidden_dim, 512),
                                       nn.Tanh(),
                                       nn.Linear(512, 512),
                                       nn.BatchNorm1d(512),
                                       nn.Tanh())    
        self.hr = nn.Sequential(nn.Linear(task_v_size, task_v_size),
                                nn.Tanh())
        self.vr = nn.Sequential(nn.Linear(2048, task_v_size),
                                nn.Tanh())
                    

        self.produce_tag_i = nn.Sequential(nn.Linear(300, task_v_size),
                                           nn.BatchNorm1d(task_v_size),
                                           nn.Tanh(),)
        self.produce_att_i = nn.Sequential(nn.Linear(300, task_v_size),
                                           nn.BatchNorm1d(task_v_size),
                                           nn.Tanh(),)
        self.produce_cnt_i = nn.Sequential(nn.Linear(self.cnt_num_class+1, task_v_size),
                                           nn.BatchNorm1d(task_v_size),
                                           nn.Tanh(),)
        self.produce_rel_i = nn.Sequential(nn.Linear(task_v_size, task_v_size),
                                           nn.BatchNorm1d(task_v_size),
                                           nn.Tanh(),)
        self.produce_cap_i = nn.Sequential(nn.Linear(2048, task_v_size),
                                           nn.BatchNorm1d(task_v_size),
                                           nn.Tanh(),)
        self.produce_vqa_i = nn.Sequential(nn.Linear(512, 2048),
                                           nn.BatchNorm1d(2048),
                                           nn.Tanh(),) 



    def forward(self, input=None, ml=None, boxes=None, attention=None, i_ns=None, wvecs=None, q_vec=None, X_loc=None, mode='', graphs=[]):
        '''
        if mode == task_embedding, give task embdding vector 
        else run controller one step
        '''
        if mode == 'get_rel_input':
            rel_emb = ml['TaskModuleRel'](mode='get_rel_emb')
            rel_emb = rel_emb.unsqueeze(0).repeat(q_vec.size(0), 1, 1)
            gate_objsub = self.gate_objsub(q_vec)
            gate_rel = self.gate_rel(q_vec)
            _, _, _, rel_logits, _ = self.rel_rel_attention(rel_emb, q_vec)
            rel_query = rel_logits

            return None, rel_query, gate_objsub, gate_rel
        elif mode == 'get_vqa_input':

            ht, icap = input
            
            r = self.hr(ht) * self.vr(icap)
            hv_gate = self.gate_hv(r)
            vqa_q = self.vqa_input(r)
            return vqa_q, hv_gate, r
       

       
        elif mode == 'attention':
            _, _, _, map_logits, _ = self.cap_img_attention(X_loc, q_vec)
            return map_logits        
        elif mode =='get_i_n':
            vecs = []
            for i_n, kind in i_ns:
                vecs.append(i_n)
            i_n_gate = self.i_n_gate(q_vec)
            i_n_stacked = torch.stack(vecs, dim=1)
            i_n = (i_n_gate.view(-1, self.num_tasks, 1) * i_n_stacked).sum(1) 
            return i_n, i_n_gate
        elif mode =='gate_hv':
            hv_gate = self.gate_hv(q_vec)
            return hv_gate


        elif mode == 'execute':
            v_cnt = ml['TaskModuleCnt'](boxes=boxes, attention=attention, mode='cnt_module', graphs=graphs)
            return ml['TaskModuleCnt'](v_cnt=v_cnt, mode='cnt'), v_cnt
        elif mode == 'produce_tag_i':
            return self.produce_tag_i(input)
        elif mode == 'produce_att_i':
            return self.produce_att_i(input)
        elif mode == 'produce_cnt_i':
            return self.produce_cnt_i(input) 
        elif mode == 'produce_rel_i':
            return self.produce_rel_i(input) 
        elif mode == 'produce_cap_i':
            return self.produce_cap_i(input) 
        elif mode == 'produce_vqa_i':
            return self.produce_vqa_i(input) 
        else:
            print 'Wrong mode to CAPController'
            exit(-1)



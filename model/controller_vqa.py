import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from attention import ImgAttention
cuda_available = torch.cuda.is_available()


class VQAController(nn.Module):
    def __init__(self, X_loc_size, K, num_tasks, input_size, task_v_size, num_hidden, num_layers, r_n_dim, att_dim, num_rel_classes, cnt_num_class, vqa_q_len, normalizer='', do_dropout=True, cap_wemb=None, cap_vocab_size=1):
        super(VQAController, self).__init__()
        self.name = 'VQAController'
        self.K = K
        self.num_tasks = num_tasks
        self.input_size = input_size
        self.num_hidden = num_hidden
        self.task_v_size = task_v_size
        self.r_n_dim = r_n_dim
        self.normalizer = normalizer
        self.X_loc_size = X_loc_size
        self.cnt_num_class = cnt_num_class
        self.vqa_q_len = vqa_q_len
        self.att_dim = att_dim
        self.cap_vocab_size = cap_vocab_size

        self.knowledge_producer = nn.GRU(task_v_size, task_v_size, 1, dropout=0.0, batch_first=True)
        self.query_producer = nn.GRU(task_v_size, task_v_size, 1, dropout=0.0, batch_first=True)

        self.wvec_attention = ImgAttention(task_v_size, task_v_size, task_v_size/2, vqa_q_len, att_mode='softmax', do_dropout=do_dropout)
        self.att_task = nn.Linear(self.task_v_size*2, 1)
        self.num_rel_classes = num_rel_classes
        self.i_n_gate = nn.Sequential(nn.Linear(task_v_size, num_tasks),
                                      nn.Softmax())
        self.cap_gate = nn.Sequential(nn.Linear(task_v_size*2, 1),
                                      nn.Sigmoid())

        self.tmp_gate = nn.Sequential(nn.Linear(task_v_size, 1))

        self.wvec_size = task_v_size
        self.cap_attention = ImgAttention(task_v_size, 300, task_v_size, 60, att_mode='softmax', do_dropout=do_dropout)
        self.rel_rel_attention = ImgAttention(self.wvec_size, task_v_size, task_v_size, num_rel_classes, att_mode='softmax', do_dropout=do_dropout)
        self.wvec_rel_obj_attention = ImgAttention(task_v_size, self.wvec_size, task_v_size/2, vqa_q_len, att_mode='softmax', do_dropout=do_dropout)
        self.wvec_rel_rel_attention = ImgAttention(task_v_size, self.wvec_size, task_v_size/2, vqa_q_len, att_mode='softmax', do_dropout=do_dropout)
        self.gate_objsub = nn.Sequential(nn.Linear(task_v_size, 2),)

        self.cap_wembed = nn.Embedding(self.cap_vocab_size, 300)
        if cap_wemb is not None:
            self.cap_wembed.weight.data.copy_(torch.from_numpy(cap_wemb))

        if do_dropout:
            self.produce_cap_i = nn.Sequential(nn.Linear(300, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),
                                               nn.Dropout(),)
            self.produce_tag_i = nn.Sequential(nn.Linear(300, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),
                                               nn.Dropout(),)
            self.produce_att_i = nn.Sequential(nn.Linear(300, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),
                                               nn.Dropout(),)
            self.produce_cnt_i = nn.Sequential(nn.Linear(cnt_num_class+1, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),
                                               nn.Dropout(),)
            self.produce_vqa_i = nn.Sequential(nn.Linear(task_v_size, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),
                                               nn.Dropout(),)
            self.produce_rel_i = nn.Sequential(nn.Linear(task_v_size, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),
                                               nn.Dropout(),)
            self.query_norm = None
        else:
            self.produce_cap_i = nn.Sequential(nn.Linear(300, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),)
                                            #   nn.Dropout(),)
            self.produce_tag_i = nn.Sequential(nn.Linear(300, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),)
                                            #   nn.Dropout(),)
            self.produce_att_i = nn.Sequential(nn.Linear(300, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),)
                                             #  nn.Dropout(),)

            self.produce_cnt_i = nn.Sequential(nn.Linear(cnt_num_class+1, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),)
                                             #  nn.Dropout(),)
            self.produce_vqa_i = nn.Sequential(nn.Linear(task_v_size, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),)
                                             #  nn.Dropout(),)
            self.produce_rel_i = nn.Sequential(nn.Linear(task_v_size, task_v_size),
                                               nn.BatchNorm1d(task_v_size),
                                               nn.Tanh(),)
                                             #  nn.Dropout(),)
            self.query_norm = None

    def forward(self, input, rel_vec=None, X_loc=None, obj_vec=None, subj_vec=None, rel_emb=None,
                att=None, Q=None, q_vec=None, wvecs=None, wembs=None, f=None, q_hidden=None,
                ml=None,  mode='task_embedding', do_wvec=False):
        '''
        '''
        if mode == 'importance_function':
            i_n_gate = self.i_n_gate(input)
            return i_n_gate
            
        elif mode == 'knowledge':
            i_n_gate, i_ns = input
            total_num_tasks = self.num_tasks
            i_n_stacked = torch.stack(i_ns, dim=1)
            sum_vec = (i_n_gate.view(-1, total_num_tasks, 1) * i_n_stacked).sum(1)
            return sum_vec

        elif mode == 'state_update':
            k_t = input
            query_input = k_t
            query_input = query_input.view(-1, self.task_v_size).unsqueeze(1)
            q_t, q_hidden = self.query_producer(query_input, q_hidden)
            q_t = q_t.squeeze(1)
            return q_t, q_hidden

        elif mode == 'get_rel_input':
            if do_wvec:
                # bidirec, attention over words
                sub_query = None
                wvecs = wvecs.permute(1,0,2)
                vqa_q_len = wvecs.size(1)
                rel_emb = ml['TaskModuleRel'](mode='get_rel_emb')
                rel_emb = rel_emb.unsqueeze(0).repeat(q_vec.size(0), 1, 1)

                _, _, rel_obj_wvec_attention, _, _ = self.wvec_rel_obj_attention(wvecs, q_vec)
                _, _, rel_rel_wvec_attention, _, _ = self.wvec_rel_rel_attention(wvecs, q_vec)
                gate_objsub = self.gate_objsub(q_vec)

                rel_obj_qm_t = (wvecs * rel_obj_wvec_attention.view(-1, vqa_q_len, 1)).sum(1)
                rel_rel_qm_t = (wvecs * rel_rel_wvec_attention.view(-1, vqa_q_len, 1)).sum(1)

                _, _, _, rel_logits, _ = self.rel_rel_attention(rel_emb, rel_rel_qm_t)
                rel_query = rel_logits
                _, _, _, obj_logits, _ = ml['ImgAttention'](X_loc, rel_obj_qm_t)
                obj_query = obj_logits

                return obj_query, rel_query, sub_query, gate_objsub, rel_obj_wvec_attention, rel_rel_wvec_attention
            else:
                sub_query = None
                rel_emb = ml['TaskModuleRel'](mode='get_rel_emb')
                rel_emb = rel_emb.unsqueeze(0).repeat(q_vec.size(0), 1, 1)

                gate_objsub = self.gate_objsub(q_vec)

                _, _, _, rel_logits, _ = self.rel_rel_attention(rel_emb, q_vec)
                rel_query = rel_logits
                _, _, _, obj_logits, _ = ml['ImgAttention'](X_loc, q_vec)
                obj_query = obj_logits

                return obj_query, rel_query, sub_query, gate_objsub, None, None
        elif mode == 'cap_attention':
            inds = input.type(torch.LongTensor)
            inds = inds.cuda(0)
            cap_emb = self.cap_wembed(inds)
            cap_vec,cap_map,_, cap_logits, _ = self.cap_attention(cap_emb, q_vec)
            return cap_vec, cap_map
        elif mode == 'receiver_cap':
            return self.produce_cap_i(input)
        elif mode == 'receiver_residual':
            return self.produce_vqa_i(input)
        elif mode == 'receiver_tag':
            return self.produce_tag_i(input)
        elif mode == 'receiver_att':
            return self.produce_att_i(input)
        elif mode == 'receiver_cnt':
            return self.produce_cnt_i(input)
        elif mode == 'receiver_rel':
            return self.produce_rel_i(input)
        else:
            print 'Wrong mode to controller'
            exit(-1)

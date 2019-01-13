import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch.autograd import Variable
from attention import ImgAttention
cuda_available = torch.cuda.is_available()

class CNTController(nn.Module):
    def __init__(self, K, task_v_size, f_dim, num_rel_classes, att_dim, vqa_q_len):
        super(CNTController, self).__init__()
        self.name = 'CNTController' 
        self.K = K
        self.task_v_size = task_v_size
        self.f_dim = f_dim
        self.num_rel_classes = num_rel_classes
        self.att_dim = att_dim
        self.vqa_q_len = vqa_q_len
        self.wvec_size = task_v_size
        self.rel_rel_attention = ImgAttention(self.wvec_size, task_v_size, task_v_size, num_rel_classes, att_mode='softmax')
        self.wvec_rel_obj_attention = ImgAttention(task_v_size, self.wvec_size, task_v_size/2, vqa_q_len, att_mode='softmax')
        self.wvec_rel_rel_attention = ImgAttention(task_v_size, self.wvec_size, task_v_size/2, vqa_q_len, att_mode='softmax')
        self.gate_objsub = nn.Sequential(nn.Linear(task_v_size, 2),)
        self.gate_rel = nn.Sequential(nn.Linear(task_v_size, 2),)
        self.wvec_size = task_v_size
        self.wvec_cnt_attention = ImgAttention(task_v_size, self.wvec_size, task_v_size/2, vqa_q_len, att_mode='softmax')
         
    def forward(self, input, ml=None, boxes=None, attention=None, wvecs=None, q_vec=None, X_loc=None, mode='', graphs=[], do_wvec=False):
        '''
        if mode == task_embedding, give task embdding vector 
        else run controller one step
        '''
        if mode == 'parse_query':
            if do_wvec:
                # bidirec, wvec attention
                wvecs = wvecs.permute(1,0,2)
                _, _, wvec_attention, _, _ = self.wvec_cnt_attention(wvecs, q_vec)
                qc_t = (wvecs * wvec_attention.view(-1, self.vqa_q_len, 1)).sum(1)
                #qc_t = q_vec
                _, _, _, cnt_logits, _ = ml['ImgAttention'](X_loc, qc_t)
                cnt_map = F.sigmoid(cnt_logits)
                gate_rel = self.gate_rel(q_vec)
                return qc_t, cnt_map, cnt_logits, wvec_attention, gate_rel
            else: 
                _, _, _, cnt_logits, _ = ml['ImgAttention'](X_loc, q_vec)
                cnt_map = F.sigmoid(cnt_logits)
                gate_rel = self.gate_rel(q_vec)
                return None, cnt_map, cnt_logits, None, gate_rel
        elif mode == 'get_rel_input':
            if do_wvec:
                # bidirec, wvec attention
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
        elif mode == 'execute':
            v_cnt = ml['TaskModuleCnt'](boxes=boxes, attention=attention, mode='cnt_module', graphs=graphs)
            return ml['TaskModuleCnt'](v_cnt=v_cnt, mode='cnt'), v_cnt
        else:
            print 'Wrong mode to CNTController'
            exit(-1)



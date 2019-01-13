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

'''
beam search code modified from:
https://github.com/ruotianluo/ImageCaptioning.pytorch
'''

class TaskModuleCap(nn.Module):
    def __init__(self, K, cap_hidden_dim, cap_emb_dim, vocab_size, cap_att_dim, X_loc_dim):
        super(TaskModuleCap, self).__init__()
        self.name = 'TaskModuleCap'
        self.num_glimpse = 1
        self.K = K
        self.cap_hidden_dim = cap_hidden_dim
        self.cap_emb_dim = cap_emb_dim
        self.vocab_size = vocab_size
        self.cap_att_dim = cap_att_dim
        self.X_loc_dim = X_loc_dim 
        self.seq_length = 16 

        task_v_size = 512
        self.task_v_size = task_v_size
        self.first_layer_input_dim = self.cap_emb_dim + X_loc_dim + self.cap_hidden_dim
        self.second_layer_input_dim = self.cap_hidden_dim + 1000 # self.X_loc_dim# + self.task_v_size
        self.bot_rnn = nn.GRU(self.first_layer_input_dim, self.cap_hidden_dim, 1, dropout=0.5, batch_first=True)
        self.top_rnn = nn.GRU(self.second_layer_input_dim, self.cap_hidden_dim, 1, dropout=0.5, batch_first=True)
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.cap_emb_dim),
                                   nn.ReLU(),
                                   nn.Dropout())    
        self.output = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.cap_hidden_dim, self.vocab_size+1),
        )

    def forward(self, vec=None, ind=None, cap_input=None, hidden=None, h_2_t=None, mode='step'):
        if mode == 'embedding':
            ind = ind.type(torch.LongTensor)
            if cuda_available:
                ind = ind.cuda(0)
            xt = self.embed(ind)
            return xt
        elif mode == 'bot_rnn': 
            input = cap_input.view(-1, self.first_layer_input_dim).unsqueeze(1)
            output, hidden = self.bot_rnn(input, hidden)
            return output.squeeze(1), hidden
        elif mode == 'top_rnn': 
            input = cap_input.view(-1, self.second_layer_input_dim).unsqueeze(1)
            output, hidden = self.top_rnn(input, hidden)
            return output.squeeze(1), hidden
        elif mode == 'output':
            output = self.output(vec)
            return output
            #output = F.log_softmax(self.output(vec)) 
            #return output

    def get_logprobs_state(self, it, state, h_2_t, tmp_box, tmp_X, tmp_X_obj, tmp_X_att, tmp_X_rel, tmp_X_loc, tmp_mean_vec, ml, overall_task):
        # 'it' is Variable contraining a word index

        cout, h_2_t, i_n_gate, hv_gate, vqa_init_maps, init_maps, rel_maps, gate_rel, cap_bot_hidden, cap_top_hidden, tmp_mean_vec = cap_step(ml, state[0], state[1], tmp_mean_vec, box=tmp_box, ind=it, h_2_t=h_2_t, overall_task=overall_task, X=tmp_X, X_loc=tmp_X_loc, X_obj=tmp_X_obj, X_att=tmp_X_att, X_rel=tmp_X_rel)
        state = [cap_bot_hidden, cap_top_hidden]
        logprobs = cout
        return logprobs, state, h_2_t, i_n_gate, init_maps, rel_maps, gate_rel, tmp_mean_vec

    def sample_beam(self, ml, X, beam_size, overall_task, box):
        batch_size = X.size(0)

        X = get_init_features(X, ml, 'cap', overall_task)
        X_loc, X_obj, X_att, X_rel = X, None, None, None
        if 'tag' in overall_task or 'vqa' in overall_task:
            X_obj = ml['TaskModuleTag'](X, have_maps=False, mode='reduce_img')
            X_loc = torch.cat([X_loc, X_obj], dim=2)
        if 'att' in overall_task or 'vqa' in overall_task:
            X_att = ml['TaskModuleAtt'](X, have_maps=False, mode='reduce_img')
            X_loc = torch.cat([X_loc, X_att], dim=2)
        if 'rel' in overall_task or 'vqa' in overall_task:
            X_rel = ml['TaskModuleRel'](boxes=box, X_loc=torch.cat([X,X_obj,X_att], dim=2), mode='get_f')
        mean_vec = torch.mean(X, dim=1)


        keep_beam = beam_size
        seq = torch.LongTensor(batch_size, keep_beam, self.seq_length).zero_()
        seqLogprobs = torch.FloatTensor(batch_size, keep_beam, self.seq_length)
        
        if 'vqa' in overall_task:
            num_task = 5
        else:
            num_task = len(overall_task.split(','))
        
        seq_i_n_gate = torch.FloatTensor(batch_size, keep_beam, self.seq_length, num_task)
        seq_init_maps = torch.FloatTensor(batch_size, keep_beam, self.seq_length, 36)
        seq_rel_maps = torch.FloatTensor(batch_size, keep_beam, self.seq_length, 36)
        seq_gate_rel = torch.FloatTensor(batch_size, keep_beam, self.seq_length, 2)

        self.done_beams = [[] for _ in range(batch_size)]
        tmp_X_obj, tmp_X_rel, tmp_X_att = None, None, None
        for k in range(batch_size):
            tmp_X = X[k:k+1].expand(beam_size, X.size(1), X.size(2))
            if 'tag' in overall_task or 'vqa' in overall_task:
                tmp_X_obj = X_obj[k:k+1].expand(beam_size, X_obj.size(1), X_obj.size(2))
            if 'att' in overall_task or 'vqa' in overall_task: 
                tmp_X_att = X_att[k:k+1].expand(beam_size, X_att.size(1), X_att.size(2))
            if 'rel' in overall_task or 'vqa' in overall_task:
                tmp_X_rel = X_rel[k:k+1].expand(beam_size, X_rel.size(1), X_rel.size(2))
            tmp_X_loc = X_loc[k:k+1].expand(beam_size, X_loc.size(1), X_loc.size(2))
            tmp_mean_vec = mean_vec[k:k+1].expand(beam_size, mean_vec.size(1))
            tmp_box = box[k:k+1].expand(beam_size, box.size(1), box.size(2))
            it = Variable(torch.Tensor([0]*beam_size), requires_grad=False).cuda(0)
            cout, h_2_t, i_n_gate, hv_gate, vqa_init_maps, init_maps, rel_maps, gate_rel, cap_bot_hidden, cap_top_hidden, tmp_mean_vec = cap_step(ml, None, None, tmp_mean_vec, box=tmp_box, ind=it, overall_task=overall_task, X=tmp_X, X_loc=tmp_X_loc, X_obj=tmp_X_obj, X_att=tmp_X_att, X_rel=tmp_X_rel)
            state = [cap_bot_hidden, cap_top_hidden]
            logprobs = cout
            
            self.done_beams[k] = self.beam_search(state, logprobs, h_2_t, i_n_gate, gate_rel, init_maps, rel_maps, beam_size, tmp_box, tmp_X, tmp_X_obj, tmp_X_att, tmp_X_rel, tmp_X_loc, tmp_mean_vec, ml, overall_task)
            for l in range(keep_beam):
                seq[k, l, :] = self.done_beams[k][l]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, l, :] = self.done_beams[k][l]['logps']
                if 'rel' in overall_task:# or 'vqa' in overall_task:
                    seq_i_n_gate[k, l, :] = self.done_beams[k][l]['i_n_gate']
                    seq_init_maps[k, l, :] = self.done_beams[k][l]['init_maps']
                    seq_rel_maps[k, l, :] = self.done_beams[k][l]['rel_maps']
                    seq_gate_rel[k, l, :] = self.done_beams[k][l]['gate_rel']

        # return the samples and their log likelihoods
        return seq, seqLogprobs, seq_i_n_gate, seq_init_maps, seq_rel_maps, seq_gate_rel 

    
    def beam_search(self, state, logprobs, h_2_t, i_n_gate, gate_rel, init_maps, rel_maps, beam_size, tmp_box, tmp_X, tmp_X_obj, tmp_X_att, tmp_X_rel, tmp_X_loc, tmp_mean_vec, ml, overall_task):
        # args are the miscelleous inputs to the core in addition to embedded word and state
        # kwargs only accept opt
        is_lstm = False
        if type(state[0]) is tuple:
            is_lstm = True
        def beam_step(logprobsf, h_2_t, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, i_n_gatef, init_mapsf, rel_mapsf, gate_relf,
                    beam_i_n_gate, beam_init_maps, beam_rel_maps, beam_gate_rel, state, overall_task):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True) # sort descending order
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            bot_hidden = state[0]
            top_hidden = state[1]
            if is_lstm:
                state = [bot_hidden[0], bot_hidden[1], top_hidden[0], top_hidden[1]]
            new_state = [_.clone() for _ in state] 
            new_h_2_t = h_2_t.clone()
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
                if 'rel' in overall_task:# or 'vqa' in overall_task:                
                    beam_i_n_gate_prev = beam_i_n_gate[:t].clone()
                    beam_init_maps_prev = beam_init_maps[:t].clone()
                    beam_rel_maps_prev = beam_rel_maps[:t].clone()
                    beam_gate_rel_prev = beam_gate_rel[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                    if 'rel' in overall_task:# or 'vqa' in overall_task:                
                        beam_i_n_gate[:t, vix] = beam_i_n_gate_prev[:, v['q']]
                        beam_init_maps[:t, vix] = beam_init_maps_prev[:, v['q']]
                        beam_rel_maps[:t, vix] = beam_rel_maps_prev[:, v['q']]
                        beam_gate_rel[:t, vix] = beam_gate_rel_prev[:, v['q']]
            
                #rearrange recurrent states
                #  copy over state in previous beam q to new beam at vix
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
                if 'rel' in overall_task:# or 'vqa' in overall_task:                
                    beam_i_n_gate[t, vix] = i_n_gatef[v['q'], :]
                    beam_init_maps[t, vix] = init_mapsf[v['q'], :]
                    beam_rel_maps[t, vix] = rel_mapsf[v['q'], :]
                    beam_gate_rel[t, vix] = gate_relf[v['q'], :]
                new_h_2_t[vix, :] = h_2_t[v['q'], :] 
            state = new_state
            h_2_t = new_h_2_t
            return beam_seq, beam_seq_logprobs, beam_logprobs_sum, beam_i_n_gate, beam_init_maps, beam_rel_maps, beam_gate_rel, h_2_t, state, candidates

        # start beam search
        beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
        beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
        beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
        if 'rel' in overall_task:# or 'vqa' in overall_task:
            beam_i_n_gate = torch.FloatTensor(self.seq_length, beam_size, i_n_gate.size(1)).zero_()
            beam_init_maps = torch.FloatTensor(self.seq_length, beam_size, init_maps.size(1)).zero_()
            beam_rel_maps = torch.FloatTensor(self.seq_length, beam_size, rel_maps.size(1)).zero_()
            beam_gate_rel = torch.FloatTensor(self.seq_length, beam_size, gate_rel.size(1)).zero_()
        else:
            beam_rel_maps, beam_gate_rel, beam_i_n_gate, beam_init_maps = None, None, None, None
        
        done_beams = []

        for t in range(self.seq_length):
            """pem a beam merge. that is,
            for every previous beam we now many new possibilities to branch out
            we need to resort our beams to maintain the loop invariant of keeping
            the top beam_size most likely sequences."""
            logprobsf = logprobs.data.float() # lets go to CPU for more efficiency in indexing operations
            if 'rel' in overall_task:# or 'vqa' in overall_task:
                i_n_gatef = i_n_gate.data.float()
                init_mapsf = init_maps.data.float()
                rel_mapsf = rel_maps.data.float()
                gate_relf = gate_rel.data.float()
            else:
                i_n_gatef, init_mapsf, rel_mapsf, gate_relf = None, None, None, None
            # suppress UNK tokens & pad token in the decoding
            logprobsf[:, 2] =  logprobsf[:, 2] - 1000  
            logprobsf[:, logprobsf.size(1)-1] =  logprobsf[:, logprobsf.size(1)-1] - 1000  
        
            a = beam_step(logprobsf,
                                        h_2_t,
                                        beam_size,
                                        t,
                                        beam_seq,
                                        beam_seq_logprobs,
                                        beam_logprobs_sum,
                                        i_n_gatef,
                                        init_mapsf,
                                        rel_mapsf,
                                        gate_relf,
                                        beam_i_n_gate,
                                        beam_init_maps,
                                        beam_rel_maps,
                                        beam_gate_rel,
                                        state, overall_task)
            beam_seq,\
            beam_seq_logprobs,\
            beam_logprobs_sum,\
            beam_i_n_gate,\
            beam_init_maps,\
            beam_rel_maps,\
            beam_gate_rel,\
            h_2_t,\
            state,\
            candidates_divm = a
            
            if is_lstm:
                state = [(state[0], state[1]), (state[2], state[3])]
            for vix in range(beam_size):
                # if time's up... or if end token is reached then copy beams
                if beam_seq[t, vix] == 1 or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[:, vix].clone(), 
                        'logps': beam_seq_logprobs[:, vix].clone(),
                        'p': beam_logprobs_sum[vix],
                        'i_n_gate': None if beam_i_n_gate is None else beam_i_n_gate[:, vix].clone(),
                        'init_maps': None if beam_rel_maps is None else beam_init_maps[:, vix].clone(),
                        'rel_maps': None if beam_rel_maps is None else beam_rel_maps[:, vix].clone(),
                        'gate_rel': None if beam_gate_rel is None else beam_gate_rel[:, vix].clone()
                    }
                    done_beams.append(final_beam)
                    # don't continue beams from finished sequences
                    beam_logprobs_sum[vix] = -1000

            # encode as vectors
            it = beam_seq[t]
            logprobs, state, h_2_t, i_n_gate, init_maps, rel_maps, gate_rel, tmp_mean_vec = self.get_logprobs_state(Variable(it.cuda()), state, h_2_t, tmp_box, tmp_X, tmp_X_obj, tmp_X_att, tmp_X_rel, tmp_X_loc, tmp_mean_vec, ml, overall_task)

        done_beams = sorted(done_beams, key=lambda x: -x['p'])[:beam_size]
        return done_beams

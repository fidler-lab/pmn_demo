import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *


def run_vqa(X, X_obj, X_att, ml, q_t, q_hidden, box, wvecs=None, do_wvec=False):
    """
    Run VQA module
    """

    # Importance function
    G = ml['VQAController'](q_t, mode='importance_function')
    
    # counting module + attention module; note, for efficiency, the attention modules of vqa and counting are shared
    tmp = run_cnt(X, X_obj, X_att, q_t, ml, box, wvecs, do_wvec=do_wvec)
    cnt_output, v_cnt, map_logits, gate_rel_logits, X_loc, cnt_map = tmp['cnt'], tmp['cnt_v'], tmp['map_logits'], tmp['gate_rel_logits'], tmp['X_loc'], tmp['cnt_map']
    v_cnt = ml['VQAController'](v_cnt, mode='receiver_cnt')

    # relationship detection module; query producer for rel is shared between vqa and cnt
    init_maps = F.softmax(map_logits.view(-1, ml['ImgAttention'].K))
    gate_rel = F.softmax(gate_rel_logits)
    rel_query_producer = 'CNTController'
    obj_query, rel_query, _, gate_objsub_logits, obj_wvec_a, rel_wvec_a = ml[rel_query_producer](None, ml=ml, X_loc=X_loc, q_vec=q_t, wvecs=wvecs, mode='get_rel_input', do_wvec=do_wvec)
    tmp = run_rel(X, X_obj, X_att, ml, box, F.softmax(obj_query), F.softmax(rel_query), None, wvecs, q_t)
    rel_output, X_rel = tmp['rel'], tmp['X_rel']
    y_rel_obj, y_rel_sub = rel_output
    gate_objsub = F.softmax(gate_objsub_logits)
    rel_map = (torch.stack([y_rel_obj.squeeze(), y_rel_sub.squeeze()], dim=1) * gate_objsub.view(-1, 2, 1)).sum(1)
    rel_map_act = F.softmax(rel_map)
    gm, gr = torch.split(gate_rel, 1, dim=1)
    maps = init_maps * gm + rel_map_act * gr
    v_rel = (maps.view(-1, ml['ImgAttention'].K, 1) * X_rel).sum(1)
    v_rel = ml['VQAController'](v_rel, mode='receiver_rel')

    # object classification module
    tmp = run_tag(X, ml, maps=maps)
    y_obj, v_obj = tmp['tag'], tmp['tag_v']
    v_obj = ml['VQAController'](v_obj, mode='receiver_tag')
    tag_output = y_obj

    # attribute classification module
    tmp = run_att(X, ml, maps=maps)
    y_att, v_att = tmp['att'], tmp['att_v']
    v_att = ml['VQAController'](v_att, mode='receiver_att')
    att_output = y_att

    # residual module
    v_t = (maps.view(-1, ml['ImgAttention'].K, 1) * X).sum(1)
    v_vqa = ml['TaskModuleVqa'](v_head=v_t, mode='v_vqa')
    v_vqa = ml['VQAController'](v_vqa, mode='receiver_residual')


    # gather outputs (V) based on their importance scores (G)
    V = [v_rel, v_cnt, v_vqa, v_obj, v_att]
    k_t = ml['VQAController']([G, V], mode='knowledge')
    # State update function
    q_t, q_hidden = ml['VQAController'](k_t, q_hidden=q_hidden, mode='state_update')



    output = {}
    output['vqa'] = k_t
    output['i_n_gate'] = G
    output['tag'] = tag_output
    output['att'] = att_output
    output['cnt'] = cnt_output
    output['rel'] = rel_output
    output['cnt_map'] = cnt_map
    output['rel_map'] = rel_map
    output['init_map'] = init_maps
    output['gate_rel'] = gate_rel
    output['rel_query'] = rel_query
    output['obj_query'] = obj_query
    output['gate_rel_logit'] = gate_rel_logits
    output['gate_objsub_logit'] = gate_objsub_logits
    output['gate_objsub'] = gate_objsub

    return q_t, q_hidden, output


def run_tag(X, ml, maps=None):
    """
    Run object classification module
    """
    X_obj = ml['TaskModuleTag'](X, have_maps=False, mode='reduce_img')
    if maps is not None:
        v_obj = (maps.view(-1, ml['ImgAttention'].K, 1) * X_obj).sum(1)
        tag_output = ml['TaskModuleTag'](v_obj, mode='obj')
    else:
        v_obj = None
        tag_output = ml['TaskModuleTag'](X_obj, mode='obj')
    return {'tag': tag_output, 'tag_v': v_obj}


def run_att(X, ml, maps=None):
    """
    Run attribute classification module
    """
    X_att = ml['TaskModuleAtt'](X, have_maps=False, mode='reduce_img')
    if maps is not None:
        v_att = (maps.view(-1, ml['ImgAttention'].K, 1) * X_att).sum(1)
        att_output = ml['TaskModuleAtt'](v_att, mode='att')
    else:
        v_att = None
        att_output = ml['TaskModuleAtt'](X_att, mode='att')
    return {'att': att_output, 'att_v': v_att}


def cap_step(ml, cap_bot_hidden, cap_top_hidden, mean_vec, cap=None, ind=None, h_2_t=None,
            overall_task='', X=None, X_loc=None, X_obj=None, X_att=None, cstep=0):
    """
    One step of captioning module
    """
    i_ns = []
    if h_2_t is None:
        h_2_t = Variable(torch.zeros(mean_vec.size(0), ml['TaskModuleCap'].cap_hidden_dim), requires_grad=False)

    # bottom state updater
    ind_t = cap[:, cstep].clone() if ind is None else ind
    ind_emb = ml['TaskModuleCap'](ind=ind_t, mode='embedding')
    cap_bot_input = torch.cat([ind_emb, mean_vec, h_2_t], dim=1)
    h_1_t, cap_bot_hidden = ml['TaskModuleCap'](cap_input=cap_bot_input, hidden=cap_bot_hidden,  mode='bot_rnn')

    # attention module
    map_logits = ml['CAPController'](X_loc=X_loc, q_vec=h_1_t, mode='attention')
    map_logits = map_logits.view(-1, ml['TaskModuleCap'].K)
    init_maps = F.softmax(map_logits)
    maps = init_maps

    # residual module
    v_cap = (maps.view(-1, ml['TaskModuleCap'].K, 1) * X).sum(1)
    i_cap = ml['CAPController'](input=v_cap, mode='produce_cap_i')
    visual_feature = i_cap
    i_ns.append([i_cap, 'cap'])

    if 'tag' in overall_task: # run object classifier
        v_obj = (maps.view(-1, ml['TaskModuleCap'].K, 1) * X_obj).sum(1)
        i_tag = ml['CAPController'](input=v_obj, mode='produce_tag_i')
        i_ns.append([i_tag, 'tag'])
    if 'att' in overall_task: # run attribute classifier
        v_att = (maps.view(-1, ml['TaskModuleCap'].K, 1) * X_att).sum(1)
        i_att = ml['CAPController'](input=v_att, mode='produce_att_i')
        i_ns.append([i_att, 'att'])

    # importance function, gather knowledge
    if len(i_ns) > 1:
        i_n, i_n_gate = ml['CAPController'](i_ns=i_ns, q_vec=h_1_t, mode='get_i_n')
    else:
        i_n = i_ns[0][0]
    visual_feature = i_n

    # top state updater
    cap_top_input = torch.cat([h_1_t, visual_feature], dim=1)
    h_2_t, cap_top_hidden = ml['TaskModuleCap'](cap_input=cap_top_input, hidden=cap_top_hidden,  mode='top_rnn')
    cout = ml['TaskModuleCap'](vec=h_2_t, mode='output')

    cout = F.log_softmax(cout)

    return cout, h_2_t, cap_bot_hidden, cap_top_hidden, mean_vec


def run_cap(X, X_obj, X_att, ml, cap, overall_task):
    """
    Run captioning module
    """
    cap_output = []
    cap_bot_hidden, cap_top_hidden, h_2_t = None, None, None

    # Precomputed
    # X_obj = ml['TaskModuleTag'](X, have_maps=False, mode='reduce_img') # object classification
    # X_att = ml['TaskModuleAtt'](X, have_maps=False, mode='reduce_img') # attribute classification
    X_loc = torch.cat([X, X_obj, X_att], dim=2)
    mean_vec = torch.mean(X_loc, dim=1)

    for cstep in range(ml['TaskModuleCap'].seq_length-1):
        cout, h_2_t, cap_bot_hidden, cap_top_hidden, mean_vec = cap_step(ml, cap_bot_hidden, cap_top_hidden,
            mean_vec, cap=cap, h_2_t=h_2_t, overall_task=overall_task, X=X, X_loc=X_loc, X_obj=X_obj,
             X_att=X_att, cstep=cstep)
        cap_output.append(cout)
    return {'cap': cap_output}


def run_rel(X, X_obj, X_att, ml, box, rel_obj, rel_rel, rel_sub, wvecs, q_t):
    """
    Run relationship detection module
    """

    # Precomputed
    # X_obj = ml['TaskModuleTag'](X, have_maps=False, mode='reduce_img') # object classification
    # X_att = ml['TaskModuleAtt'](X, have_maps=False, mode='reduce_img') # attribute classification
    X_loc = torch.cat([X,X_obj,X_att], dim=2)
    X_rel = ml['TaskModuleRel'](boxes=box, X_loc=X_loc, mode='get_f')
    rel_output = ml['RELController'](X_rel=X_rel, rel_obj=rel_obj, rel_rel=rel_rel, rel_sub=rel_sub,
                                    wvecs=wvecs, q_vec=q_t, ml=ml, X_loc=X_loc, mode='execute')
    return {'rel': rel_output, 'X_rel': X_rel}


def run_cnt(X, X_obj, X_att, q_t, ml, box, wvecs, do_wvec=False):
    """
    Run counting module
    """

    # Precomputed
    # X_obj = ml['TaskModuleTag'](X, have_maps=False, mode='reduce_img') # object classification
    # X_att = ml['TaskModuleAtt'](X, have_maps=False, mode='reduce_img') # attribute classification
    X_loc = torch.cat([X, X_obj, X_att], dim=2)

    # attention module
    qc_t, init_maps, sig_logits, wvec_a, gate_rel_logits = ml['CNTController'](None, ml=ml, wvecs=wvecs, q_vec=q_t, X_loc=X_loc, mode='parse_query', do_wvec=do_wvec)
    gate_rel = F.softmax(gate_rel_logits)

    # produce input for relationship module and run it
    obj_query, rel_query, _, gate_objsub_logits, obj_wvec_a, rel_wvec_a = ml['CNTController'](None, ml=ml, X_loc=X_loc, q_vec=q_t, wvecs=wvecs, mode='get_rel_input', do_wvec=do_wvec)
    gate_objsub = F.softmax(gate_objsub_logits)
    tmp = run_rel(X, X_obj, X_att, ml, box, F.softmax(obj_query), F.softmax(rel_query), None, wvecs, q_t)
    rel_output, X_rel = tmp['rel'], tmp['X_rel']
    y_rel_obj, y_rel_sub = rel_output
    rel_map = (torch.stack([y_rel_obj.view(-1, 36), y_rel_sub.view(-1,36)], dim=1) * gate_objsub.view(-1, 2, 1)).sum(1)
    rel_map_sigmoid = F.sigmoid(rel_map)

    # importance function, choose output (attention module vs relationship module)
    gm, gr = torch.split(gate_rel, 1, dim=1)
    maps = init_maps * gm  + rel_map_sigmoid * gr

    cnt_output, v_cnt = ml['CNTController'](None, ml=ml, boxes=box[:,:,:4], attention=maps, mode='execute')
    return {'cnt': cnt_output, 'cnt_v': v_cnt, 'map_logits': sig_logits, 'gate_rel_logits': gate_rel_logits, 'X_loc': X_loc, 'cnt_map': maps}



def run_model(inputs, ml, dicts, log_dir, cur_task='tag', overall_task='tag,cnt,att,vqa,rel',
              optimizer=None, step=0, train=True, epoch=0, do_wvec=False):
    '''
    '''
    vqa_criterion = nn.BCEWithLogitsLoss()
    cap_criterion = LanguageModelCriterion()
    if 'vqa' == cur_task:
        x, question, y, box, datapath, q_id = inputs
        V_cur, wvecs, wembs = ml['QuestionEmbedding'](q_inds=question, mode='question_vec')
    elif 'cap' == cur_task:
        x, box, cap, mask, img_ids, rnd_const = inputs
    elif 'tag_first' == cur_task or 'att_first' == cur_task:
        x, y = inputs
        obj_label, att_label, use_label, att_use_label, box, datapath = y
    elif 'cnt' == cur_task:
        x, question, cnt_label,  box, use_label, datapath, train_attn, attn_target, cur_num_boxes, rnd_const = inputs
        V_cur, wvecs, wembs = ml['QuestionEmbedding'](q_inds=question, mode='question_vec')
    elif 'rel' == cur_task:
        x, box, rel_obj, rel_rel, rel_sub, rel_obj_y, rel_sub_y, rnd_const, objs, datapath = inputs

    k_hidden, q_hidden = None, None
    loss, final_loss = 0, 0
    X = get_init_features(x, cur_task)

    if 'vqa' == cur_task:
        # pre-execute object and attribute classification modules for efficiency
        X_obj = ml['TaskModuleTag'](X, have_maps=False, mode='reduce_img')  # object classification
        X_att = ml['TaskModuleAtt'](X, have_maps=False, mode='reduce_img')  # attribute classification
        # state initializer
        q_t, q_hidden = ml['VQAController'](V_cur, q_hidden=None, mode='state_update')

    vqa_outputs, cap_outputs, tag_outputs, att_outputs, cnt_outputs, rel_outputs, \
    cur_atts, other_atts, attentions, rel_maps, gate_rels, gate_objsubs, rel_querys, \
    i_n_gates, obj_querys,  k_ts, gate_rel_logits, gate_objsub_logits, \
    cap_maps, cnt_maps = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # For num_steps, run a task module
    num_steps = ml['TaskModule'+cur_task[0].upper()+cur_task[1:3]].num_glimpse-1
    for att_step_i in range(num_steps):
        if cur_task == 'tag_first':
            output = run_tag(X, ml)
        elif cur_task == 'att_first':
            output = run_att(X, ml)
        elif cur_task == 'rel':
            output = run_rel(X, X_obj, X_att, ml, box, rel_obj, rel_rel, rel_sub, wvecs, q_t)
        elif cur_task == 'cap':
            output = run_cap(X, ml, overall_task)
        elif cur_task == 'cnt':
            output = run_cnt(X, ml, overall_task)
        elif cur_task == 'vqa':
            q_t, q_hidden, output = run_vqa(X, X_obj, X_att, ml, q_t, q_hidden,
                                            box, wvecs=wvecs, do_wvec=do_wvec)
    
        vqa_outputs = add_output(vqa_outputs, 'vqa', output)
        cap_outputs = add_output(cap_outputs, 'cap', output)
        tag_outputs = add_output(tag_outputs, 'tag', output)
        att_outputs = add_output(att_outputs, 'att', output)
        rel_outputs = add_output(rel_outputs, 'rel', output)
        cnt_outputs = add_output(cnt_outputs, 'cnt', output)
        i_n_gates = add_output(i_n_gates, 'i_n_gate', output)
        rel_querys = add_output(rel_querys, 'rel_query', output)
        obj_querys = add_output(obj_querys, 'obj_query', output)
        gate_rel_logits = add_output(gate_rel_logits, 'gate_rel_logit', output)
        gate_objsub_logits = add_output(gate_objsub_logits, 'gate_objsub_logit', output)
        gate_objsubs = add_output(gate_objsubs, 'gate_objsub', output)
        rel_maps = add_output(rel_maps, 'rel_map', output)
        cap_maps = add_output(cap_maps, 'cap_map', output)
        cnt_maps = add_output(cnt_maps, 'cnt_map', output)
        attentions = add_output(attentions, 'init_map', output)

    if 'vqa' in cur_task:
        vqa_output, vqa_vecs = ml['TaskModuleVqa'](k_ts=vqa_outputs, Q=V_cur, mode='output')
        vqa_loss = vqa_criterion(vqa_output, y)
        vqa_loss *= y.size(1)
        obj_output, att_output = None, None
        if len(tag_outputs) > 0:
            obj_output, att_output = [], []
            for o_out, a_out in zip(tag_outputs, att_outputs):
                if len(obj_output) == num_steps-1:
                    break
                obj_output.append(o_out)
                att_output.append(a_out)

        X_obj_label = ml['TaskModuleTag'](X_obj, mode='obj')
        X_att_label = ml['TaskModuleAtt'](X_att, mode='att')
        write_vqa(question, vqa_output, y, attentions, obj_output, att_output, cnt_outputs, log_dir, box, datapath, dicts, X_obj_label, X_att_label,
                 rel_maps, gate_rel_logits, rel_querys, obj_querys, i_n_gates, gate_objsubs, cnt_maps, q_id)

        final_loss += vqa_loss
        out = vqa_output

    elif 'cap' in cur_task:
        cap_x = torch.cat([_.unsqueeze(1) for _ in cap_outputs[0]], dim=1)
        cap_loss = cap_criterion(cap_x, cap[:,1:], mask[:,1:])
        final_loss += cap_loss
        out = cap_x

    elif 'tag' in cur_task:
        obj_output = tag_outputs[0]
        ce = nn.CrossEntropyLoss()
        obj_outputs = torch.split(obj_output, 1)
        obj_labels = torch.split(obj_label.view(-1), 1)
        assert len(obj_outputs) == len(obj_labels)
        obj_losses = []
        for i in range(len(obj_outputs)):
            obj_losses.append(ce(obj_outputs[i], obj_labels[i]))
        obj_loss = torch.cat(obj_losses, dim=0)
        obj_loss = obj_loss * use_label.view(-1)
        obj_loss = torch.sum(obj_loss)/torch.sum(use_label)

        final_loss += obj_loss
        out = [obj_output, None, maps, obj_loss.data[0], 0]

    elif 'att' in cur_task:
        att_output = att_outputs[0]
        bce = nn.BCEWithLogitsLoss()
        att_outputs = torch.split(att_output, 1)
        att_labels = torch.split(att_label.view(att_label.size(0), -1), 1)
        assert len(att_outputs) == len(att_labels)
        att_losses = []
        for i in range(len(att_outputs)):
            att_losses.append(bce(att_outputs[i], att_labels[i]))
        att_loss = torch.cat(att_losses, dim=0)
        att_loss = att_loss * att_use_label.view(-1)
        att_loss = torch.sum(att_loss)/torch.sum(att_use_label)

        final_loss += att_loss
        out = [None, att_output, maps, 0, att_loss.data[0]]
    elif 'cnt' in cur_task:
        cnt_output = cnt_outputs[0]
        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss()

        cnt_loss = ce(cnt_output, cnt_label)

        sig_logits_all = torch.split(sig_logits, 1)
        attn_target_all = torch.split(attn_target.view(attn_target.size(0), -1), 1)
        assert len(sig_logits_all) == len(attn_target_all)
        var_losses = []
        for i in range(len(sig_logits_all)):
            var_loss = bce(sig_logits_all[i], attn_target_all[i])
            var_losses.append(var_loss)
        var_loss = torch.cat(var_losses, dim=0)
        var_loss = var_loss * train_attn.view(-1)
        var_loss = torch.sum(var_loss)/torch.sum(train_attn)

        final_loss += cnt_loss
        out = [cnt_output, cnt_loss.data[0], var_loss.data[0]]
    elif 'rel' in cur_task:
        rel_output_obj, rel_output_sub = rel_outputs[0]
        bce = nn.BCEWithLogitsLoss()

        rel_output_obj = rel_output_obj.view(-1, ml['TaskModuleRel'].K)
        rel_loss_obj = bce(rel_output_obj, rel_sub)
        rel_output_sub = rel_output_sub.view(-1, ml['TaskModuleRel'].K)
        rel_loss_sub = bce(rel_output_sub, rel_obj)

        final_loss += (rel_loss_obj  + rel_loss_sub) * rel_obj.size(1)
        out = [rel_output_obj, rel_output_sub, rel_loss_obj.data[0], rel_loss_sub.data[0]]

    if train:
        final_loss.backward()

        if cur_task == 'cap':
            clip_gradient(optimizer, 10.0)
        else:
            clip_gradient(optimizer, 0.25)

    return final_loss, out

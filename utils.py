import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.metrics import average_precision_score
from torch.autograd import Variable
from shutil import copyfile
from data.data import MSCOCOvqa
import time
import cv2

cuda_available = torch.cuda.is_available()
coco_orig_path = '/ais/gobi6/seung/seung-project/mscoco'

def vis_detections(im, class_name, dets, thresh=0.8, highest=False, use_colour=None, h=1.0, w=1.0, rc=1.0, alpha=1.0):
    """Visual debugging of detections."""
    overlay = im
    for i in range(dets.shape[0]):
        overlay = im.copy()
        output = im.copy()
        bbox = []
        for x, ind in zip(dets[i, :4], range(4)):
            if ind == 0 or ind == 2:
                x = int(np.round(x * w / rc))
            if ind == 1 or ind ==3:
                x = int(np.round(x * h / rc))
            bbox.append(x)
        bbox = tuple(bbox)
        thickness = 2
        if highest:
            colour = (0, 0, 200)
        else:
            colour = (0, 200, 0)
        if use_colour is not None:
            colour = use_colour

        if alpha == 1.0:
            cv2.rectangle(overlay, bbox[0:2], bbox[2:4], colour, thickness=thickness)
            cv2.putText(overlay, '%s' % (class_name), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 255), thickness=1)
            return overlay
        else:
            cv2.rectangle(overlay, bbox[0:2], bbox[2:4], colour, -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            return output
    return overlay

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1).float()
        output = -input.gather(1, target.long()) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr #param_group['lr']*0.1

def calculate_tag_stat(all_labels, all_preds, all_preds_top3, num_correct, num_error, val_loss, num_val_tag, num_seen, total_samples, t0, epoch):
    tmp_labels = []
    for ent in all_labels:
        for lb in ent:
            tmp_labels.append(lb)
    tmp_preds = []
    for ent in all_preds:
        for pr in ent:
            tmp_preds.append(pr)
    tmp_preds_top3 = []
    for ent in all_preds_top3:
        for pr in ent:
            tmp_preds_top3.append(pr)
    all_labels = np.stack(tmp_labels, axis=0)
    all_preds = np.stack(tmp_preds, axis=0)
    all_preds_top3 = np.stack(tmp_preds_top3, axis=0)

    map_score = average_precision_score(np.copy(all_labels), np.copy(all_preds))
    map_score_top3 = average_precision_score(np.copy(all_labels), np.copy(all_preds_top3))
    accuracy = num_correct*1.0/(num_seen)
    num_error = num_error*1.0/(total_samples)
    val_loss = val_loss/num_val_tag
    map_score2 = calculate_mAP(np.copy(all_labels), np.copy(all_preds))
    map_score2_top3 = calculate_mAP(np.copy(all_labels), np.copy(all_preds_top3))
    log_str = 'Epoch(val): ' + str(epoch) + ',\tLoss: ' + str(val_loss) + \
                   ',\tnum_error: ' + str(num_error) + '\n' + \
                   '\tmAP: ' + str(map_score) + ',\tmAP2: ' + str(map_score2) + '\n' + \
                   '\tmAP(top3): ' + str(map_score_top3) + ',\tmAP2(top3):  ' + str(map_score2_top3) +'(' +str( "%.2f" % (time.time()-t0)) + ' sec) \n'

    return log_str, map_score2

def calculate_mAP(labels, probs):
    num_cls = len(labels[0])
    AP_all = np.zeros(num_cls)
    for m in range(num_cls):
        gt = labels[:, m]
        out = probs[:, m]
        si = np.argsort(out)[::-1]
        tp = gt[si]
        fp = (-1.0*(gt-1.0))[si]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp/sum(gt)
        prec = tp/(fp+tp)
        ap = calculate_AP(rec, prec)
        AP_all[m] = ap
    return np.mean(AP_all)

def calculate_AP(rec, prec):
    mrec = np.insert(rec, 0, 0)
    mrec = np.insert(mrec, len(mrec), 1)
    mpre = np.insert(prec, 0, 0)
    mpre = np.insert(mpre, len(mpre), 0)
    for i in range(len(mpre)-2, 0, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i = np.nonzero(mrec[1:]-mrec[:-1])[0]+1
    ap = sum((mrec[i]-mrec[i-1]) * mpre[i])
    return ap

def get_model(chk, name):
    if chk.has_key(name):
        return chk[name]
    print '##### checkpoint does not have module: ' + name
    return None

def to_variable_cuda(variables, kind, volatile=False):
    for v in range(len(variables)):
        if kind[v] == 'float':
            if cuda_available:
                variables[v] = Variable((variables[v].type(torch.FloatTensor)).cuda(0), requires_grad=False, volatile=volatile)
            else:
                variables[v] = Variable((variables[v].type(torch.FloatTensor)), requires_grad=False, volatile=volatile)
        else:
            if cuda_available:
                variables[v] = Variable((variables[v].type(torch.LongTensor)).cuda(0), requires_grad=False, volatile=volatile)
            else:
                variables[v] = Variable((variables[v].type(torch.LongTensor)), requires_grad=False, volatile=volatile)

    return variables

def to_np(x):
    '''
    cast to numpy
    '''
    return x.data.cpu().numpy()

def get_dictionary(path):
    dicts = pickle.load(open(path, 'rb'))
    return  dicts

def get_dataloader(vqa_data, path, batch_size, kwargs, w2i):

    loader = torch.utils.data.DataLoader(
                MSCOCOvqa(vqa_data, path, w2i, num_boxes=36),
                batch_size=batch_size, shuffle=False, drop_last=False, **kwargs)
    return loader

def set_mode(module_lst, mode, cuda_ind=0):
    '''
    set model mode: 'zero_grad', 'train', 'eval', 'cuda'
    '''
    for name, mod in module_lst.iteritems():
        if mode == 'zero_grad':
            mod.zero_grad()
        elif mode == 'train':
            mod.train()
        elif mode == 'eval':
            mod.eval()
        elif mode == 'cuda':
            mod.cuda(cuda_ind)
        else:
            print 'not implemented'
            exit(-1)


def write_vqa(question, vqa_x, y, attentions, obj_output, att_output, cnt_outputs, log_dir, box, datapath, dicts, X_obj_label_orig, X_att_label_orig, \
              rel_maps, gate_rels, rel_querys, obj_querys,  i_n_gates, gate_objsubs, cnt_maps, q_id):
    '''
    write prediction classes something
    '''
    power = 4
    vqa_x = F.softmax(vqa_x)
    num_sample = vqa_x.size(0)
    rc = 1.0

    for sample_ind in range(num_sample):
        info = {}
        cur_q = q_id[sample_ind]
        img_path = datapath[sample_ind]
        img_id = img_path.split('/')[-1].split('.')[0]
        file_name = '0' * (12 - len(str(img_id))) + str(img_id) + '.jpg'
        file_name = 'COCO_val2014_'+file_name
        orig_img_path = os.path.join(coco_orig_path, 'val2014', file_name)
        copyfile(orig_img_path, os.path.join(log_dir, str(cur_q)+'_'+str(img_id)+'_orig.jpg'))
        cur_boxes = box[sample_ind].data.cpu().numpy()
        X_obj_label = X_obj_label_orig.data[sample_ind].cpu().numpy()
        X_att_label = X_att_label_orig.data[sample_ind].cpu().numpy()

        att_step = 0 # analyze the first time step
        im = cv2.imread(orig_img_path)
        h, w, _ = im.shape

        ######################### Output of attention module ##########################
        im2show = np.copy(im)
        cur_attentions = attentions[att_step][sample_ind].data.cpu().numpy()
        assert len(cur_attentions) == len(cur_boxes)
        topind = np.argmax(cur_attentions)

        im2show = vis_detections(im2show, '', np.array([cur_boxes[topind]]), 0.0, use_colour=(0, 0, 200), h=h, w=w,
                                 rc=rc, alpha=0.4)
        im2show = vis_detections(im2show, '', np.array([cur_boxes[topind]]), 0.0, use_colour=(0, 0, 200), h=h, w=w,
                                 rc=rc)
        cv2.imwrite(os.path.join(log_dir, str(cur_q)+'_'+str(img_id) + '_attention.jpg'), im2show)
        obj_label = dicts['obj_i2w'][np.argmax(X_obj_label[topind])]
        a_out = X_att_label[topind].argsort()[-3:][::-1]
        att_label = ''
        for a in a_out:
            att_label += str(dicts['att_i2w'][a])
        info['vqa_obj_name'] = obj_label
        info['vqa_att_name'] = att_label
        ################################################################################

        ##################### input & output for relationship module #####################
        info['objsub'] = gate_objsubs[att_step][sample_ind].data.cpu().numpy()
        info['rel_map_same'] = False

        topind_input = np.argmax(obj_querys[att_step][sample_ind].data.cpu().numpy())
        im2show = np.copy(im)
        im2show = vis_detections(im2show, '', np.array([cur_boxes[topind_input]]), 0.0, use_colour=(200, 0, 0), h=h,
                                 w=w, rc=rc, alpha=0.4)
        im2show = vis_detections(im2show, '', np.array([cur_boxes[topind_input]]), 0.0, use_colour=(200, 0, 0), h=h,
                                     w=w, rc=rc)
        obj_label = dicts['obj_i2w'][np.argmax(X_obj_label[topind_input])]
        a_out = X_att_label[topind_input].argsort()[-3:][::-1]
        att_label = ''
        for a in a_out:
            att_label += str(dicts['att_i2w'][a])
        info['rel_input_obj_name'] = obj_label
        info['rel_input_att_name'] = att_label

        topind_rel = np.argmax(F.softmax(rel_maps[att_step])[sample_ind].data.cpu().numpy())
        im2show = vis_detections(im2show, '', np.array([cur_boxes[topind_rel]]), 0.0, use_colour=(0, 200, 0), h=h, w=w, rc=rc, alpha=0.4)
        im2show = vis_detections(im2show, '', np.array([cur_boxes[topind_rel]]), 0.0, use_colour=(0, 200, 0), h=h, w=w, rc=rc)

        obj_label = dicts['obj_i2w'][np.argmax(X_obj_label[topind_rel])]
        a_out = X_att_label[topind_rel].argsort()[-3:][::-1]
        att_label = ''
        for a in a_out:
            att_label += str(dicts['att_i2w'][a])
        info['rel_output_obj_name'] = obj_label
        info['rel_output_att_name'] = att_label
        cv2.imwrite(os.path.join(log_dir, str(cur_q)+'_'+str(img_id) + '_rel.jpg'), im2show)
        if topind_input == topind_rel:
            info['rel_map_same'] = True
        ################################################################################

        ################################ counting map ##################################
        info['cnt_map_blank'] = False
        im2show = np.copy(im)
        cur_cnt_maps = np.power(cnt_maps[att_step][sample_ind].data.cpu().numpy(), power)
        cnt_inds = np.where(cur_cnt_maps > 0.5)
        tmp = cur_cnt_maps
        topind = np.argmax(tmp)
        alphas = tmp/tmp[topind] * 0.5
        info['cnt_obj_name'] = []
        info['cnt_att_name'] = []
        for ind in range(len(cur_boxes)): #, alphas):
            cb, alpha = cur_boxes[ind], alphas[ind]
            im2show = vis_detections(im2show, '', np.array([cb]), 0.0, use_colour=(100, 0, 150), h=h, w=w, rc=rc, alpha=alpha)
            if alpha > 0.2:
                obj_label = dicts['obj_i2w'][np.argmax(X_obj_label[ind])]
                a_out = X_att_label[ind].argsort()[-3:][::-1]
                att_label = ''
                for a in a_out:
                    att_label += str(dicts['att_i2w'][a])
                info['cnt_obj_name'].append(obj_label)
                info['cnt_att_name'].append(att_label)
        cv2.imwrite(os.path.join(log_dir, str(cur_q)+'_'+str(img_id) + '_cnt.jpg'), im2show)
        if len(cnt_inds) == 0:
            info['cnt_map_blank'] = True
        ################################################################################

        #### Question ####
        q = question.data[sample_ind].cpu().numpy()
        q_str = ''
        for q_ind in q:
            if q_ind < len(dicts['vqa_q_i2w']):
                q_str += dicts['vqa_q_i2w'][q_ind] + ' '
            else:
                break
        ##################

        predy = vqa_x.data[sample_ind].cpu().numpy()
        predy_ind_total = np.argmax(predy)
        cls_str = str(sample_ind) + ': '
        ny = y.data[sample_ind].cpu().numpy()
        cls_inds = np.nonzero(ny)[0]
        gt_ans = []
        for cl in cls_inds:
            cls_str += dicts['vqa_a_i2w'][cl] + '(' +  str(ny[cl]) + '), '
            gt_ans.append([dicts['vqa_a_i2w'][cl], ny[cl]])
        info['question'] = q_str
        info['gt_ans'] = gt_ans
        info['pred_ans'] = [dicts['vqa_a_i2w'][predy_ind_total], predy[predy_ind_total]]
        rel_max_ind = np.argmax(rel_querys[att_step][sample_ind].data.cpu().numpy())
        info['gate_rel'] = [gate_rels[att_step][sample_ind].data.cpu().numpy(), dicts['rel_i2w'][rel_max_ind]]
        info['i_n_gate'] = i_n_gates[att_step][sample_ind].data.cpu().numpy()
        cnt_max_ind = np.argmax(cnt_outputs[att_step][sample_ind].data.cpu().numpy())
        info['count'] = str(cnt_max_ind)

        o_out = np.argmax(obj_output[0].data[sample_ind].cpu().numpy())
        info['obj'] = dicts['obj_i2w'][o_out]
        a_out = att_output[0].data[sample_ind].cpu().numpy().argsort()[-3:][::-1]
        att_l = []
        for cl in a_out:
            att_l.append(dicts['att_i2w'][cl])
        info['att'] = att_l

        pickle.dump(info, open(os.path.join(log_dir, str(cur_q)+'_'+str(img_id)+'_info.p'), 'wb'))
    return


def write_summary(slist, step, lg, log_file):
    '''
    write summary for given modules
    '''
    print 'writing summary'
    for s, kind in slist:
        if kind == 'histogram':
            hist_summary(s, lg, step, log_file, name=s.name)
        elif kind == 'scalar':
            name, val = s
            lg.scalar_summary(name, val, step)
    print 'summary written'

def add_output(l, key, output_list):
    if output_list.has_key(key):
        l.append(output_list[key])
    return l


def get_init_features(x, cur_task):
    '''
    initial featrue for controller
    '''
    cnn_features = x
    if 'first' in cur_task:
        cnn_features = F.normalize(cnn_features, dim=1)
    else:
        cnn_features = F.normalize(cnn_features, dim=2)
    return cnn_features

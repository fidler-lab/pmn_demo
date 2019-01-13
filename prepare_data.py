from data import construct_tag_dict
import pickle

obj_w2i,obj_i2w,att_w2i,att_i2w,rel_w2i,rel_i2w = construct_tag_dict('/ais/gobi6/seung/seung-project/faster-rcnn.pytorch/data/genome/1600-400-20')
vqa_trn_data, vqa_val_data, vqa_q_i2w, vqa_q_w2i, vqa_a_i2w, vqa_a_w2i = pickle.load(open('coco_vqa_data.p', 'rb'))
vqa_captions = pickle.load(open('/ais/gobi6/seung/seung-project/multitask/log_cap/cap_base_mv_continue/eval16/coco_cap_extracted.p', 'rb'))
cap_trn_data, cap_trn_ids, cap_val_data, cap_val_ids, cap_w2i, cap_i2w = pickle.load(open('coco_cap_data.p', 'rb'))

dicts = {}
dicts['obj_w2i'] = obj_w2i
dicts['att_w2i'] = att_w2i
dicts['rel_w2i'] = rel_w2i
dicts['vqa_q_w2i'] = vqa_q_w2i
dicts['vqa_a_w2i'] = vqa_a_w2i
dicts['cap_w2i'] = cap_w2i

dicts['obj_i2w'] = obj_i2w
dicts['att_i2w'] = att_i2w
dicts['rel_i2w'] = rel_i2w
dicts['vqa_q_i2w'] = vqa_q_i2w
dicts['vqa_a_i2w'] = vqa_a_i2w
dicts['cap_i2w'] = cap_i2w

pickle.dump(dicts, open('dicts.p', 'wb'))
pickle.dump([vqa_val_data, vqa_captions], open('vqa_val_cap.p', 'wb'))
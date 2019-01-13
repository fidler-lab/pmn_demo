import torch.utils.data as data
import os,sys
import numpy as np
import pickle
sys.path.insert(0, '../')

def default_loader(path):
    return pickle.load(open(path, 'rb'))

def parse_data(data, cur_num_boxes, w, h, num_boxes):
    features, boxes, attn_target, use, objs, atts, att_use = [], [], [], [], [], [], []
    for i in range(cur_num_boxes):
        cb, cf, co, ca = data['boxes'][i], data['features'][i], [], []
        features.append(np.asarray(cf))
        boxes.append(np.asarray([cb[0]*1.0/w, cb[1]*1.0/h, cb[2]*1.0/w, cb[3]*1.0/h, ((cb[2]-cb[0])*(cb[3]-cb[1])*1.0)/(w*h)]))

    pad_len = num_boxes - cur_num_boxes
    for i in range(pad_len):
        features.append(np.asarray([0.0]*2048))
        boxes.append(np.asarray([0.0]*5))
    return features, boxes
 

class MSCOCOvqa(data.Dataset):
    def __init__(self, data, path, w2i, num_boxes=36, q_len=14, loader=default_loader):
        self.data = data
        self.path = path
        self.loader = loader
        self.max_len = q_len
        self.a_vocab_size = len(w2i[0])
        self.q_vocab_size = len(w2i[1])
        self.num_boxes = num_boxes
   
    
    def __getitem__(self, index):
        cur_data = self.data[index]
        img_id = cur_data['image_id']
        question = cur_data['question'][:self.max_len]
        question_id = -1
        if cur_data.has_key('question_id'):
            question_id = cur_data['question_id']
        pad_len = max(0, self.max_len-len(question))
        question = question + [self.q_vocab_size]*pad_len
        answers = cur_data['answers']

        data_path = os.path.join(self.path, str(img_id)+'.pkl')
        try:
            data = self.loader(data_path)
        except:
            print('error in loading pkl from ' + str(data_path))
            exit(-1)
        cur_num_boxes = data['num_boxes']
        w = data['image_w']
        h = data['image_h']

        features, boxes = parse_data(data, cur_num_boxes, w, h, self.num_boxes)
        
        label = np.zeros(self.a_vocab_size)
        for ans in answers:
            w, c = ans
            label[w] = float(c)

        return np.asarray(features, dtype=np.float32), np.asarray(boxes, dtype=np.float32), np.asarray(question), \
               label, question_id, data_path
    
    def __len__(self):
        return len(self.data)

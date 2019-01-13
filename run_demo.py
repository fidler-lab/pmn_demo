import json
import numpy as np
import torch
from utils import *
from model_runner import run_model
import argparse

parser = argparse.ArgumentParser(description='module')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--log_dir', type=str, default='test', metavar='N',
                    help='log directory')
parser.add_argument('--resume_path', type=str, default='model_ckpt.t7', metavar='N',
                    help='path to saved model')
parser.add_argument('--dict_path', type=str, default='data/dicts.p', metavar='N',
                    help='path to saved dictionaries')
parser.add_argument('--data_path', type=str, default='data/demo_data.p', metavar='N',
                    help='path to data file')
parser.add_argument('--img_path', type=str, default='', metavar='N',
                    help='path to img file')
parser.add_argument('--wvec', action='store_true', default=False,
                    help='do wvec attention')
args = parser.parse_args()
kwargs = {'num_workers': 15, 'pin_memory': torch.cuda.is_available()}
cuda_available = torch.cuda.is_available()
checkpoint = torch.load(args.resume_path)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

def step_vqa(iter_vqa, ml, dicts, train=True):
    x, box, question, y, q_id, datapath = next(iter_vqa)
    y = y.type(torch.FloatTensor)
    x, box, question, y = to_variable_cuda([x, box, question, y], ['float', 'float', 'long', 'float'], volatile=True)
    inputs = [x, question, y, box, datapath, q_id]
    loss, pred_ans = run_model(inputs, ml, dicts, args.log_dir, cur_task='vqa',train=train, do_wvec=args.wvec)
    return loss, pred_ans, q_id


def run(ml):
    vqa_data = pickle.load(open(args.data_path, 'rb'))
    dicts = get_dictionary(args.dict_path)
    print("\ndata loaded.\n")

    set_mode(ml, 'eval')
    preds, q_ids = [], []
    val_vqa_loader = get_dataloader(vqa_data,  args.img_path, args.batch_size, kwargs, \
                                    [dicts['vqa_a_w2i'], dicts['vqa_q_w2i']])
    val_iter_vqa = iter(val_vqa_loader)
    num_steps = len(val_iter_vqa)
    total_loss = 0
    for i in range(num_steps):
        loss, pred_ans, val_q_id = step_vqa(val_iter_vqa, ml, dicts, train=False)
        pred_ans = pred_ans.data.cpu().numpy()
        pred_ans = np.argmax(pred_ans, axis=1)
        for q_id, ans in zip(val_q_id, pred_ans):
            preds.append({'question_id': q_id, 'answer': dicts['vqa_a_i2w'][ans]})
            q_ids.append(q_id)
        total_loss += loss.data.cpu().numpy()[0]
        print(str(i)+'/'+str(num_steps) + ', ' + str(total_loss/(i+1)))


def main():
    QEmb = get_model(checkpoint, 'QuestionEmbedding')
    Imgatt = get_model(checkpoint, 'ImgAttention')
    VQAC = get_model(checkpoint, 'VQAController')
    Mvqa = get_model(checkpoint, 'TaskModuleVqa')
    CNTC = get_model(checkpoint, 'CNTController')
    Mcnt = get_model(checkpoint, 'TaskModuleCnt')
    RELC = get_model(checkpoint, 'RELController')
    Mrel = get_model(checkpoint, 'TaskModuleRel')
    Mtag = get_model(checkpoint, 'TaskModuleTag')
    Matt = get_model(checkpoint, 'TaskModuleAtt')
    module_lst = [QEmb,Imgatt,VQAC,Mvqa,CNTC,Mcnt,RELC,Mrel,Mtag,Matt]
    print('\nmodel restored.\n')
    
    ml = {}
    for m in module_lst:
        ml[m.name] = m
    if torch.cuda.is_available():
        set_mode(ml, 'cuda')

    run(ml)

if __name__ == "__main__":
    main()

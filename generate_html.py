import sys,os
import pickle

threshold = 0.3

def form_img(path):
    s = "<img src='file:///" + path + "' width='200' height='200' />\n"
    return s

def create_html(img_id, path, info):
    correct = False
    for entry in info['gt_ans']:
        if info['pred_ans'][0] == entry[0]:
            correct = True
    s = '<br/><br/><br/><div><h3>Img_id: ' + img_id + '</h3>\n'
    s += '<div>\n'
    s += '<h5>Input image and question: ' + info['question'] + '</h5><br/>\n'
    s += form_img(os.path.join(path, img_id + '_orig.jpg'))
    s += '</div><br/>\n'
    s += '<div>\n'
    s += '<h5>Reasoning process</h5>\n'

    cnt_used, rel_used = False, False
    if info['i_n_gate'][1] > threshold:
        s += form_img(os.path.join(path, img_id + '_cnt.jpg'))
        s += '<ul>\n'
        s += '<li>I look at the <strong>PURPLE</strong> boxes.</li>\n'
        s += '<li>I will try to count them: <strong>' + info['count'] + '</strong></li>\n'
        cnt_used = True
    else:
        if info['gate_rel'][0][0] > info['gate_rel'][0][1]:
            s += form_img(os.path.join(path, img_id + '_attention.jpg'))
            s += '<ul>\n'
            s += '<li>I look at the <strong>RED</strong> box.</li>\n'
        else:
            s += form_img(os.path.join(path, img_id + '_rel.jpg'))
            s += '<ul>\n'
            if info['rel_map_same']:
                s += '<li>I look at the <strong>GREEN</strong> box. </li>\n'
            else:
                s += '<li>I first find the <strong>BLUE</strong> box, and then from that'
                s += ' I look at the <strong>GREEN</strong> box. </li>\n'
            rel_used = True

    if info['i_n_gate'][3] > threshold:
        if cnt_used:
            obj_str = ' '.join([obj[0] for obj in info['cnt_obj_name']])
        elif rel_used:
            obj_str = info['rel_output_obj_name'][0]
        else:
            obj_str = info['vqa_obj_name'][0]
        s += '<li>The object(s) <strong>' + obj_str
        s += '</strong> would be useful in answering the question.</li>\n'
    if info['i_n_gate'][4] > threshold:
        if cnt_used:
            att_str = info['cnt_att_name'][0]
        elif rel_used:
            att_str = info['rel_output_att_name']
        else:
            att_str = info['vqa_att_name']
        s += '<li>The object properties <strong>' + att_str
        s += '</strong> would be useful in answering the question.</li>\n'


    s += '<li>In conclusion, I think the answer is <strong>' + info['pred_ans'][0] + '</strong></li>\n'
    s += '</ul>\n</div>\n<hr>\n'
    return s





if __name__  == '__main__':
    path = sys.argv[1]

    base_str = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\n<title>Demo results</title>\n"
    base_str += "<link rel=\"stylesheet\" href=\"css/styles.css?v=1.0\">\n"
    base_str += "<script src=\"https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js\"></script>\n"
    base_str += "</head>\n<body>\n"
    main_str = ''
    for fname in os.listdir(path):
        if fname.endswith('.p'):
            img_id = fname.split('_')[0] + '_' + fname.split('_')[1]
            cur_file = os.path.join(path, fname)
            main_str += create_html(img_id, path, pickle.load(open(cur_file, 'rb')))
    html = open(os.path.join(path, 'result.html'), 'wb')
    html.write(base_str + main_str + "</body>")
    html.close()

    sys.exit(0)

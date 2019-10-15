import os
import json
from konlpy.tag import Kkma

current_path = os.path.dirname(os.path.abspath( __file__ ))
parent_path = os.path.join(current_path, '..')

sent_gs_train_path = os.path.join(parent_path, 'data/sent_gs_train.json')
sent_jl_train_path = os.path.join(parent_path, 'data/sent_jl_train.json')
sent_gs_test_path = os.path.join(parent_path, 'data/sent_gs_test.json')
sent_jl_test_path = os.path.join(parent_path, 'data/sent_jl_test.json')
sent_gs_train_path_ex = os.path.join(parent_path, 'data/ex/sent_gs_train.json')
sent_jl_train_path_ex = os.path.join(parent_path, 'data/ex/sent_jl_train.json')

kkma = Kkma()

def make_dict(path):
    ret_dict = dict()
    with open(path) as fp:
        ret_dict = json.load(fp)
    print('\nload', path)
    return ret_dict

def save_dict(sent_dict, path):
    with open(path, 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(sent_dict, ensure_ascii=False, indent='\t'))
    print('\nsave', path)   

def extract_noun(sent_dict):
    print('\nextract_noun start!')
    count = 0
    for sent in sent_dict:
        try:
            print('\r%d/%d' % (count, len(sent_dict)), end='')
            noun_list = kkma.nouns(sent['standard'])
            std = sent['standard'].split()
            dia = sent['dialect'].split()
            for i in range(len(std)):
                last_pair = ''
                for noun in noun_list:
                    len_noun = len(noun)
                    idx = std[i].find(noun)
                    jdx = dia[i].find(noun)
                    if idx != -1 and jdx != -1:
                        std[i] = std[i][:idx] + ' ' + std[i][idx:idx+len_noun] + ' ' + std[i][idx+len_noun:]
                        dia[i] = dia[i][:jdx] + ' ' + dia[i][jdx:jdx+len_noun] + ' ' + dia[i][jdx+len_noun:]
            sent['standard'] = ' '.join(' '.join(std).split())
            sent['dialect'] = ' '.join(' '.join(dia).split())
            count = count + 1
        except:
            print(std, dia)

    print('\nextract_noun finished!')
    return sent_dict

if __name__ == "__main__":
    sent_dict = dict()

    sent_dict = make_dict(sent_gs_train_path)
    sent_dict = extract_noun(sent_dict)
    save_dict(sent_dict, sent_gs_train_path_ex)

    sent_dict = make_dict(sent_jl_train_path)
    sent_dict = extract_noun(sent_dict)
    save_dict(sent_dict, sent_jl_train_path_ex)
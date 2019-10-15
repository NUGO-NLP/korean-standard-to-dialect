import json
import os

from model.statistical import statisticalModel
from evaluate import word_accuracy, sentence_accuracy, word_accuracy_oov

current_path = os.path.dirname(os.path.abspath( __file__ ))
sent_gs_valid_filename = os.path.join(current_path, 'data/sent_gs_test.json')
sent_jl_valid_filename = os.path.join(current_path, 'data/sent_jl_test.json')
result_gs_filename = os.path.join(current_path, 'output/sent_result_gs.json')
result_jl_filename = os.path.join(current_path, 'output/sent_result_jl.json')
word_dict_gs_filename = os.path.join(current_path, 'model/save/statistical_word_dict_gs.json')
word_dict_jl_filename = os.path.join(current_path, 'model/save/statistical_word_dict_jl.json')

statistical_model_gs = statisticalModel('gs')
statistical_model_jl = statisticalModel('jl')

def test_and_write(model, input_filename, output_filename):
    sent_test = dict()
    with open(input_filename) as input_file:
        sent_test = json.load(input_file)
    with open(output_filename, 'w', encoding='UTF-8') as output_file:
        for sentence in sent_test:
            standard = sentence['standard']
            inference = model.inference_sentence(standard)
            sentence['inference'] = inference
        output_file.write(json.dumps(sent_test, ensure_ascii=False, indent='\t'))
    return sent_test

def make_sentence_list(sent_dict):
    ret_list = list()
    for sent in sent_dict:
        ret_list.append((sent['standard'], sent['dialect'], sent['inference']))
    return ret_list

def make_sentence_list_same(sent_dict):
    ret_list = list()
    for sent in sent_dict:
        ret_list.append((sent['standard'], sent['dialect'], sent['standard']))
    return ret_list

if __name__ == "__main__":
    
    sent_dict = test_and_write(statistical_model_gs, sent_gs_valid_filename, result_gs_filename)
    sent_list = make_sentence_list(sent_dict)
    print('[gs] word_accuracy:', word_accuracy(sent_list))
    print('[gs] sentence_accuracy:', sentence_accuracy(sent_list))
    with open(word_dict_gs_filename) as word_dict_gs_file:
        word_dict_gs = json.load(word_dict_gs_file)
        print('[gs] word_accuracy_oov:', word_accuracy_oov(sent_list, word_dict_gs))
    
    print()
    sent_list_same = make_sentence_list_same(sent_dict)
    print('[gs_same] word_accuracy:', word_accuracy(sent_list_same))
    print('[gs_same] sentence_accuracy:', sentence_accuracy(sent_list_same))
    with open(word_dict_gs_filename) as word_dict_gs_file:
        word_dict_gs = json.load(word_dict_gs_file)
        print('[gs_same] word_accuracy_oov:', word_accuracy_oov(sent_list_same, word_dict_gs))

    print()
    sent_dict = test_and_write(statistical_model_jl, sent_jl_valid_filename, result_jl_filename)
    sent_list = make_sentence_list(sent_dict)
    print('[jl] word_accuracy:', word_accuracy(sent_list))
    print('[jl] sentence_accuracy:', sentence_accuracy(sent_list))
    with open(word_dict_jl_filename) as word_dict_jl_file:
        word_dict_jl = json.load(word_dict_jl_file)
        print('[jl] word_accuracy_oov:', word_accuracy_oov(sent_list, word_dict_jl))
    
    print()
    sent_list_same = make_sentence_list_same(sent_dict)
    print('[jl_same] word_accuracy:', word_accuracy(sent_list_same))
    print('[jl_same] sentence_accuracy:', sentence_accuracy(sent_list_same))
    with open(word_dict_jl_filename) as word_dict_jl_file:
        word_dict_jl = json.load(word_dict_jl_file)
        print('[jl_same] word_accuracy_oov:', word_accuracy_oov(sent_list_same, word_dict_jl))

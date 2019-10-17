from collections import Counter
import numpy as np
from nltk import ngrams
import nltk.translate.bleu_score as bleu

def word_accuracy_oov(sentence_list, word_dict):
    size = 0
    count = 0
    for std, dia, inf in sentence_list:
        word_dict_std = std.split()
        word_list_dia = dia.split()
        word_list_inf = inf.split()
        for i in range(len(word_list_dia)):
            if word_dict_std[i] not in word_dict:
                #print(word_dict_std[i], word_list_dia[i], word_list_inf[i])
                size = size + 1
                if word_list_dia[i] == word_list_inf[i]:
                    count = count + 1

    return count / size

def word_accuracy(sentence_list):
    size = 0
    count = 0

    for std, dia, inf in sentence_list:
        word_list_dia = dia.split()
        word_list_inf = inf.split()
        for i in range(len(word_list_dia)):
            size = size + 1
            if word_list_dia[i] == word_list_inf[i]:
                count = count + 1

    return count / size

def sentence_accuracy(sentence_list):
    size = len(sentence_list)
    count = 0

    for std, dia, inf in sentence_list:
        if dia == inf:
            count = count + 1
    return count / size

def bleu_score(sentence_list, n_gram=4):
    weights = [1./ n_gram for _ in range(n_gram)]
    
    try:
        smt_func = bleu.SmoothingFunction()
        score = 0.0
        
        for _, dia, inf in sentence_list:
            score += bleu.sentence_bleu([dia.split()],
                                        inf.split(),
                                        weights,
                                        smoothing_function=smt_func.method2)
        if len(sentence_list) : 
            return 0
        else :
            return score / len(sentence_list)
    except Exception as ex:
        print(ex)
        return 0

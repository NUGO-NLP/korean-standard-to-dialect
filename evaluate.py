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

def bleu_score(sentence_list):
    pass
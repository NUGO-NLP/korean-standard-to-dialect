import nltk.translate.bleu_score as bleu
import torch 
import json

def translate_sentence(model, SRC, TRG, tokenized_sentence, input_level, device):
    model.eval()
    numericalized = SRC.indexesFromSentence(tokenized_sentence, sos=True, eos=True)
    tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device) 
    translation_tensor_logits = model(tensor, None, 0) 
    translation_tensor = torch.argmax(translation_tensor_logits.squeeze(1), 1)
    translation_index = [t.item() for t in translation_tensor]

    translation = TRG.sentenceFromIndex(translation_index)
    translation = translation[1:]
    if input_level == 'syl':
        translation = ''.join(translation)
    elif input_level == 'word':
        translation = ' '.join(translation)
    elif input_level == 'jaso':        
        print("NOT IMPLEMENTED!")
        exit()
        
    eos_idx = translation.find('<')
    if eos_idx != -1:
        translation = translation[:eos_idx].strip()
    return translation

def save_log(path, model, SRC, TRG, test_pair, input_level, device):
    result = []

    # srcs = test_loader.srcs[portion:]
    # trgs = test_loader.trgs[portion:]
    id = 1
    for src, trg in test_pair:
        inf = translate_sentence(model, SRC, TRG, src, input_level, device)
        data = {}
        data['id'] = id
        data['standard'] = src
        data['dialect'] = trg
        data['inference'] = inf
        result.append(data)     
        id += 1

    with open(path, 'w', encoding='UTF-8') as fp:
        fp.write(json.dumps(result, ensure_ascii=False, indent='\t'))
    print(f'save log : {path}')
    return result

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
        if len(sentence_list) == 0: 
            return 0
        else :
            return score / len(sentence_list)
    except Exception as ex:
        print(ex)
        return 0
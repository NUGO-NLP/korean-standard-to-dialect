import os
import math
import re
import json
from konlpy.tag import Kkma

current_path = os.path.dirname(os.path.abspath( __file__ ))
parent_path = os.path.join(current_path, '..')

class statisticalModel:
    def __init__(self, region):
        # Region can only be 'gs' or 'jl'
        assert region == 'gs' or region == 'jl', 'region should be \'gs\' or \'jl\''

        self.kkma = Kkma()
        self.region = region
        self.sent_dict = dict()
        self.word_dict = dict()
        self.sent_dict_subword = dict()
        self.word_dict_subword = dict()
        
        self.sentence_data_filename = os.path.join(parent_path, 'data/sent_' + region + '_train.json')
        self.word_data_filename = os.path.join(parent_path, 'data/word_' + region + '_train.json')
        self.sentence_data_ex_filename = os.path.join(parent_path, 'data/ex/sent_' + region + '_train.json')
        self.word_data_ex_filename = os.path.join(parent_path, 'data/ex/word_' + region + '_train.json')

        self.sentence_dict_filename = os.path.join(current_path, 'save/statistical_sent_dict_' + region + '.json')
        self.word_dict_filename = os.path.join(current_path, 'save/statistical_word_dict_' + region + '.json')
        self.sentence_dict_ex_filename = os.path.join(current_path, 'save/ex/statistical_sent_dict_' + region + '.json')
        self.word_dict_ex_filename = os.path.join(current_path, 'save/ex/statistical_word_dict_' + region + '.json')
        

        # If there is a dictionary created
        if os.path.isfile(self.sentence_dict_filename) and os.path.isfile(self.word_dict_filename) and \
            os.path.isfile(self.sentence_dict_ex_filename) and os.path.isfile(self.word_dict_ex_filename):
            self.load_dict()
            print('Load dictionary for %s' % self.region)
        else:
            self.create_dict()
            print('Create and load dictionary for %s' % self.region)

    def create_dict(self):
        with open(self.sentence_data_filename) as sentence_data_file:
            sentence_list = json.load(sentence_data_file)

            for sentence in sentence_list:
                new_sentence = dict()
                new_sentence['standard'] = sentence['standard']
                new_sentence['dialect'] = sentence['dialect']

                self.sent_dict[str(sentence['id'])] = new_sentence
        
        with open(self.sentence_data_ex_filename) as sentence_data_ex_file:
            sentence_list = json.load(sentence_data_ex_file)

            for sentence in sentence_list:
                new_sentence = dict()
                new_sentence['standard'] = sentence['standard']
                new_sentence['dialect'] = sentence['dialect']

                self.sent_dict_subword[str(sentence['id'])] = new_sentence

        with open(self.word_data_filename) as word_data_file:
            word_list = json.load(word_data_file)

            for word in word_list:
                sentence_id = word['sentence']
                standard = word['standard']
                dialect = word['dialect']
                
                if standard == dialect:
                    continue

                if standard not in self.word_dict:
                    self.word_dict[standard] = dict()

                if dialect not in self.word_dict[standard]:
                    self.word_dict[standard][dialect] = list()

                self.word_dict[standard][dialect].append(str(sentence_id))
        
        with open(self.word_data_ex_filename) as word_data_ex_file:
            word_list = json.load(word_data_ex_file)

            for word in word_list:
                sentence_id = word['sentence']
                standard = word['standard']
                dialect = word['dialect']
                
                if standard == dialect:
                    continue

                if len(standard) == 1:
                    continue

                if standard not in self.word_dict_subword:
                    self.word_dict_subword[standard] = dict()

                if dialect not in self.word_dict_subword[standard]:
                    self.word_dict_subword[standard][dialect] = list()

                self.word_dict_subword[standard][dialect].append(str(sentence_id))

        with open(self.sentence_dict_filename, 'w', encoding='UTF-8') as sentence_dict_file:
            sentence_dict_file.write(json.dumps(self.sent_dict, ensure_ascii=False, indent='\t'))

        with open(self.sentence_dict_ex_filename, 'w', encoding='UTF-8') as sentence_dict_ex_file:
            sentence_dict_ex_file.write(json.dumps(self.sent_dict_subword, ensure_ascii=False, indent='\t'))

        with open(self.word_dict_filename, 'w', encoding='UTF-8') as word_dict_file:
            word_dict_file.write(json.dumps(self.word_dict, ensure_ascii=False, indent='\t'))

        with open(self.word_dict_ex_filename, 'w', encoding='UTF-8') as word_dict_ex_file:
            word_dict_ex_file.write(json.dumps(self.word_dict_subword, ensure_ascii=False, indent='\t'))

    def load_dict(self):
        with open(self.sentence_dict_filename) as sentence_dict_file:
            self.sent_dict = json.load(sentence_dict_file)

        with open(self.word_dict_filename) as word_dict_file:
            self.word_dict = json.load(word_dict_file)
        
        with open(self.sentence_dict_ex_filename) as sentence_dict_ex_file:
            self.sent_dict_subword = json.load(sentence_dict_ex_file)

        with open(self.word_dict_ex_filename) as word_dict_ex_file:
            self.word_dict_subword = json.load(word_dict_ex_file)

    def inference_subword(self, word, sentence):
        max_sentence_list_len = 0
        word_infer = ''
        if word in self.word_dict_subword:
            dialect_word_dict = self.word_dict_subword[word]
            for dialect_word in dialect_word_dict.keys():
                sentence_id_list = dialect_word_dict[dialect_word]
                if len(sentence_id_list) > max_sentence_list_len:
                    max_sentence_list_len = len(sentence_id_list)
                    word_infer = dialect_word
            return word_infer
        else:
            return word

    def inference_word_by_subword(self, target, sentence):
        # Make sentences in subword units.
        word_list = sentence.split()
        noun_list = self.kkma.nouns(sentence)

        sentence_subword = ''
        for word in word_list:
            for noun in noun_list:
                len_noun = len(noun)
                idx = word.find(noun)
                if idx != -1:
                    word = word[:idx] + ' ' + word[idx:idx+len_noun] + ' ' + word[idx+len_noun:]
                    word = word.strip()
            sentence_subword = sentence_subword + ' ' + word
        
        for noun in noun_list:
            len_noun = len(noun)
            idx = target.find(noun)
            if idx != -1:
                target = target[:idx] + ' ' + target[idx:idx+len_noun] + ' ' + target[idx+len_noun:]
                target = target.strip()

        word_infer = ''
        subword_list = target.split()
        for subword in subword_list:
            subword_infer = self.inference_subword(subword, sentence_subword)
            word_infer = word_infer + subword_infer
        return word_infer

    def dot(self, x, y):
        ret = 0
        for i in range(len(x)):
            ret += x[i] * y[i]
        return ret

    def dist(self, x):
        ret = 0
        for i in range(len(x)):
            ret += x[i] * x[i]
        return math.sqrt(ret)

    # Cosine Similarity
    def get_score(self, sentence, sentence_cmp):
        word_list = sentence.split()
        word_list_cmp = sentence_cmp.split()

        word_set = list(set(word_list + word_list_cmp))
        word_dict = dict()
        for i in range(len(word_set)):
            word_dict[word_set[i]] = i

        bow = [0 for i in range(len(word_set))]
        bow_cmp = [0 for i in range(len(word_set))]

        for word in word_list:
            bow[word_dict[word]] = bow[word_dict[word]] + 1
        for word in word_list_cmp:
            bow_cmp[word_dict[word]] = bow_cmp[word_dict[word]] + 1
        
        return self.dot(bow, bow_cmp) / (self.dist(bow) * self.dist(bow_cmp))

    def inference_word(self, word, sentence='None'):
        if sentence == 'None':
            # Probably not used code.
            max_sentence_list_len = 0
            word_infer = ''
            if word in self.word_dict:
                dialect_word_dict = self.word_dict[word]
                for dialect_word in dialect_word_dict.keys():
                    sentence_id_list = dialect_word_dict[dialect_word]
                    if len(sentence_id_list) >= max_sentence_list_len:
                        max_sentence_list_len = len(sentence_id_list)
                        word_infer = dialect_word
                return word_infer
            else:
                return word
        else:
            max_score = 0
            max_sentence_list_len = 0
            word_infer = ''
            if word in self.word_dict:
                dialect_word_dict = self.word_dict[word]
                for dialect_word in dialect_word_dict.keys():
                    sentence_id_list = dialect_word_dict[dialect_word]
                    for sentence_id in sentence_id_list:
                        sentence_cmp = self.sent_dict[sentence_id]['standard']
                        score = self.get_score(sentence, sentence_cmp)
                        if score > max_score:
                            max_score = score
                            word_infer = dialect_word
                        elif score == max_score:
                            if len(sentence_id_list) >= max_sentence_list_len:
                                max_sentence_list_len = len(sentence_id_list)
                                word_infer = dialect_word
                return word_infer
            else:
                # If there are no matching words in the dictionary,
                # just returns a word.
                # You can use some neural network model for this part.
                return self.inference_word_by_subword(word, sentence)

    def inference_sentence(self, sentence):
        sentence = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', sentence)
        word_infer_list = list()
        word_list = sentence.split()

        for word in word_list:
            word_infer = self.inference_word(word, sentence)
            word_infer_list.append(word_infer)

        sentence_infer = " ".join(word_infer_list)
        return sentence_infer
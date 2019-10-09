import os
import json

current_path = os.path.dirname(os.path.abspath( __file__ ))
parent_path = os.path.join(current_path, '..')

class ruleBase:
    # region = 'gs' or 'jl'
    def __init__(self, region):
        assert region == 'gs' or region == 'jl', 'region should be \'gs\' or \'jl\''

        self.region = region
        self.sent_dict = dict()
        self.word_dict = dict()

        self.sentence_data_filename = os.path.join(parent_path, 'data/sent_' + region + '_train.json')
        self.word_data_filename = os.path.join(parent_path, 'data/word_' + region + '_train.json')
        self.sentence_dict_filename = os.path.join(current_path, 'save/rulebase_sent_dict_' + region + '.json')
        self.word_dict_filename = os.path.join(current_path, 'save/rulebase_word_dict_' + region + '.json')

        if os.path.isfile(self.sentence_dict_filename) and os.path.isfile(self.word_dict_filename):
            self.load_dict()
            print('load complete!')
        else:
            self.make_dict()
            print('make & load complete!')

    def make_dict(self):
        with open(self.sentence_data_filename) as sentence_data_file:
            sentence_list = json.load(sentence_data_file)

            for sentence in sentence_list:
                new_sentence = dict()
                new_sentence['standard_word_list'] = sentence['standard'].split()
                new_sentence['dialect_word_list'] = sentence['dialect'].split()

                self.sent_dict[str(sentence['id'])] = new_sentence

        with open(self.word_data_filename) as word_data_file:
            word_list = json.load(word_data_file)

            for word in word_list:
                sentence_id = word['sentence']
                standard = word['standard']
                dialect = word['dialect']

                if standard not in self.word_dict:
                    self.word_dict[standard] = dict()

                if dialect not in self.word_dict[standard]:
                    self.word_dict[standard][dialect] = list()

                self.word_dict[standard][dialect].append(str(sentence_id))

        with open(self.sentence_dict_filename, 'w', encoding='UTF-8') as sentence_dict_file:
            sentence_dict_file.write(json.dumps(self.sent_dict, ensure_ascii=False, indent='\t'))

        with open(self.word_dict_filename, 'w', encoding='UTF-8') as word_dict_file:
            word_dict_file.write(json.dumps(self.word_dict, ensure_ascii=False, indent='\t'))

    def load_dict(self):
        with open(self.sentence_dict_filename) as sentence_dict_file:
            self.sent_dict = json.load(sentence_dict_file)

        with open(self.word_dict_filename) as word_dict_file:
            self.word_dict = json.load(word_dict_file)

    def inference_word(self, word, sentence='None'):
        if sentence == 'None':
            pass
        else:
            pass

    def inference_sentence(self, sentence):
        word_infer_list = list()
        word_list = sentence.split()

        for word in word_list:
            word_infer = self.inference_word(word, sentence)
            word_infer_list.append(word_infer)

        sentence_infer = " ".join(word_infer_list)
        return sentence_infer
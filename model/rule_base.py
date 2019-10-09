import os
import json

current_path = os.path.dirname(os.path.abspath( __file__ ))
parent_path = os.path.join(current_path, '..')

class ruleBase:
    def __init__(self, region):
        # Region can only be 'gs' or 'jl'
        assert region == 'gs' or region == 'jl', 'region should be \'gs\' or \'jl\''

        self.region = region
        self.sent_dict = dict()
        self.word_dict = dict()

        self.sentence_data_filename = os.path.join(parent_path, 'data/sent_' + region + '_train.json')
        self.word_data_filename = os.path.join(parent_path, 'data/word_' + region + '_train.json')
        self.sentence_dict_filename = os.path.join(current_path, 'save/rulebase_sent_dict_' + region + '.json')
        self.word_dict_filename = os.path.join(current_path, 'save/rulebase_word_dict_' + region + '.json')

        # If there is a dictionary created
        if os.path.isfile(self.sentence_dict_filename) and os.path.isfile(self.word_dict_filename):
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

    def get_score(self, sentence, sentence_id_list):
        score = 0
        word_list = sentence.split()
        word_list = set(word_list)
        sentence_list_size = len(sentence_id_list)

        for sentence_id in sentence_id_list:
            target_word_list = self.sent_dict[sentence_id]['standard_word_list']
            score += len(word_list & set(target_word_list))
        
        return score / sentence_list_size

    def inference_word(self, word, sentence='None'):
        if sentence == 'None':
            # Probably not used code.
            max_sentence_list_len = 0
            word_infer = ''
            if word in self.word_dict:
                dialect_word_dict = self.word_dict[word]
                for dialect_word in dialect_word_dict.keys():
                    sentence_id_list = dialect_word_dict[dialect_word]
                    if len(sentence_id_list) > max_sentence_list_len:
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
                    score = self.get_score(sentence, sentence_id_list)
                    if score > max_score:
                        max_score = score
                        word_infer = dialect_word
                    elif score == max_score:
                        if len(sentence_id_list) > max_sentence_list_len:
                            word_infer = dialect_word
                
                return word_infer
            else:
                # If there are no matching words in the dictionary,
                # just returns a word.
                # You can use some neural network model for this part.
                return word

    def inference_sentence(self, sentence):
        word_infer_list = list()
        word_list = sentence.split()

        for word in word_list:
            word_infer = self.inference_word(word, sentence)
            word_infer_list.append(word_infer)

        sentence_infer = " ".join(word_infer_list)
        return sentence_infer
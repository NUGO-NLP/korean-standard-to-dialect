import re, json
import torch

class Vocab:
    def __init__(self, sentences, input_level, device):
        self.sentences = sentences
        self.stoi = {'<pad>': 0, '<sos>': 1, '<eos>':2, '<unk>':3}
        self.s_freq = {'<pad>': 0, '<sos>': 0, '<eos>':0, '<unk>':0}
        self.itos = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
        self.vocab_size = 4
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3
        self.input_level = input_level
        self.device = device
    
    def _addWord(self, s):
        if s not in self.stoi:
            self.stoi[s] = self.vocab_size
            self.s_freq[s] = 1
            self.itos[self.vocab_size] = s
            self.vocab_size += 1
        else:
            self.s_freq[s] += 1

    def build_vocab(self):
        for sentence in self.sentences:
            if self.input_level == 'syl':
                sentence = [ch for ch in sentence]
            elif self.input_level == 'word':
                sentence = sentence.split()
            elif self.input_level == 'jaso':
                print("NOT IMPLEMENTED!")
                exit()

            for s in sentence:
                self._addWord(s)
    
    def sentenceFromIndex(self, indexs):
        ret_list = []
        for t in indexs:
            if t in self.itos:
                ret_list.append(self.itos[t])
            else:   
                ret_list.append('<unk>')
        return ret_list

    def indexesFromSentence(self, sentence, sos, eos):
        ret_list = []
        if sos:
            ret_list.append(self.SOS_IDX)

        if self.input_level == 'syl':
            sentence = [ch for ch in sentence]
        elif self.input_level == 'word':
            sentence = sentence.split()
        elif self.input_level == 'jaso':
            print("NOT IMPLEMENTED!")
            exit()

        for s in sentence:
            if s in self.stoi:
                ret_list.append(self.stoi[s])
            else:
                ret_list.append(self.UNK_IDX)
        if eos:
            ret_list.append(self.EOS_IDX)
        return ret_list

    def tensorFromSentence(self, sentence, sos, eos):
        indexes = self.indexesFromSentence(sentence, sos, eos)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)
        # return = [sent len, batch size]
        # return = [sent len, 1]

class Loader:
    def __init__(self, max_len, input_level):
        self.pairs = []
        self.srcs = []
        self.trgs = []
        self.max_len = max_len
        self.input_level = input_level

    def _normalize(self, sentence):
        sentence = sentence.strip()
        sentence = re.sub(r"[^가-힣 ]", r"", sentence)
        return sentence
    
    def _filterPairs(self):
        filtered = []
        for pair in self.pairs:
            if self.input_level == 'syl':
                len_p0 = len(pair[0])
                len_p1 = len(pair[1])
            if self.input_level == 'word':
                len_p0 = len(pair[0].split())
                len_p1 = len(pair[1].split())

            if len_p0 < self.max_len and len_p1 < self.max_len:
                filtered.append(pair)
        self.pairs = filtered

    def readJson(self, path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        self.pairs = [[self._normalize(d['standard']), self._normalize(d['dialect'])] for d in data]
        self._filterPairs()
        self.srcs = [pair[0] for pair in self.pairs]
        self.trgs = [pair[1] for pair in self.pairs]
    
    def makeIterator(self, SRC, TRG, sos, eos):
        # If sos, eos is True, add <sos>, <eos> tokens.
        ret_list = []
        for pair in self.pairs:
            src = SRC.tensorFromSentence(pair[0], sos, eos)
            trg = TRG.tensorFromSentence(pair[1], sos, eos)
            ret_list.append([src, trg])
        return ret_list

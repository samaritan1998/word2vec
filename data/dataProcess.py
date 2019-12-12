import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import config
from config import DefaultConfig
import re

opt=DefaultConfig()
MAX_VOCAB_SIZE=opt.MAX_VOCAB_SIZE
C=opt.C
K=opt.K

class dataProcess(object):

    # tokenize函数，把一篇文本转化成一个个单词
    def word_tokenize(self,text):
        return text.split()

    def getText(self):

        with open("./data/cut_std_zh_wiki_00", "r") as fin:
            text = fin.read()

        text = [w for w in self.word_tokenize(text.lower())]
        vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
        idx_to_word = [word for word in vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
        VOCAB_SIZE = len(idx_to_word)
        return text
        #return VOCAB_SIZE
    def getWordidx(self):
        with open("./data/cut_std_zh_wiki_00", "r") as fin:
            text = fin.read()

        text = [w for w in self.word_tokenize(text.lower())]
        vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
        idx_to_word = [word for word in vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
        VOCAB_SIZE = len(idx_to_word)
        return word_to_idx
    def getIdxword(self):
        with open("./data/cut_std_zh_wiki_00", "r") as fin:
            text = fin.read()

        text = [w for w in self.word_tokenize(text.lower())]
        vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
        idx_to_word = [word for word in vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
        VOCAB_SIZE = len(idx_to_word)
        return idx_to_word
    def getVocabsize(self):
        with open("./data/cut_std_zh_wiki_00", "r") as fin:
            text = fin.read()

        text = [w for w in self.word_tokenize(text.lower())]
        vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
        idx_to_word = [word for word in vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
        VOCAB_SIZE = len(idx_to_word)
        return VOCAB_SIZE
    def getWordcounts(self):
        with open("./data/cut_std_zh_wiki_00", "r") as fin:
            text = fin.read()

        text = [w for w in self.word_tokenize(text.lower())]
        vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
        idx_to_word = [word for word in vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
        VOCAB_SIZE = len(idx_to_word)
        return word_counts
    def getWordfreq(self):
        with open("./data/cut_std_zh_wiki_00", "r") as fin:
            text = fin.read()

        text = [w for w in self.word_tokenize(text.lower())]
        print(len(text))

        vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
        idx_to_word = [word for word in vocab.keys()]
        word_to_idx = {word: i for i, word in enumerate(idx_to_word)}
        word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
        word_freqs = word_counts / np.sum(word_counts)
        word_freqs = word_freqs ** (3. / 4.)
        word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
        VOCAB_SIZE = len(idx_to_word)
        return word_freqs





class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts,VOCAB_SIZE):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        print(VOCAB_SIZE-1)
        #print(word_to_idx)
        print(word_to_idx.get('intern'))
        print(len(text))
        #print(text)
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text] #返回指定键的值，如果值不在字典中返回default值
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = dict(word_to_idx)
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(np.array(word_freqs))
        self.word_counts = torch.Tensor(np.array(word_counts))

    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        #print("dataloader也运行了")
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words

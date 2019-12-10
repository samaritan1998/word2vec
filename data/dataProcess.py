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
        # 下面这一行去掉也没什么吧，不行。因为idx+C+1可能比整个len还长
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        '''
        torch.multinomial(input, num_samples,replacement=False, out=None) → LongTensor

作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标。

输入是一个input张量，一个取样数量，和一个布尔值replacement。

input张量可以看成一个权重张量，每一个元素代表其在该行中的权重。如果有元素为0，那么在其他不为0的元素

被取干净之前，这个元素是不会被取到的。

n_samples是每一行的取值次数，该值不能大于每一样的元素数，否则会报错。

replacement指的是取样时是否是有放回的取样，True是有放回，False无放回。

看官方给的例子：
>>> weights = torch.Tensor([0, 10, 3, 0]) # create a Tensor of weights
>>> torch.multinomial(weights, 4)

 1
 2
 0
 0
[torch.LongTensor of size 4]

>>> torch.multinomial(weights, 4, replacement=True)

 1
 2
 1
 2
[torch.LongTensor of size 4]
输入是[0,10,3,0]，也就是说第0个元素和第3个元素权重都是0，在其他元素被取完之前是不会被取到的。

所以第一个multinomial取4次，可以试试重复运行这条命令，发现只会有2种结果：[1 2 0 0]以及[2 1 0 0]，以[1 2 0 0]这种情况居多。这其实很好理解，第1个元素权重比第2个元素权重要大，所以先取第1个元素的概率就会大。在第1和2个元素取完之后，剩下了2个没有权重的元素，它们才会被取到。但实际上权重为0的元素被取到时也不会显示正确的下标，关于0的下标问题我还没有想到很合理的解释，先行略过。

而第二个multinomial取4次，发现就只会出现1和2这两个元素了。这是因为replacement为真，所以有放回，就永远也不会取到权重为0的元素了。

再试试输入二维张量，则返回的也会成为一个二维张量，行数为输入的行数，列数为n_samples，即每一行都取了n_samples次，取法和一维张量相同。
        '''

        return center_word, pos_words, neg_words

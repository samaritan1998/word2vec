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


class dataProcess(object):

    # tokenize函数，把一篇文本转化成一个个单词
    def word_tokenize(text):
        return text.split()

    def getData(self):
        with open("text8.train.txt", "r") as fin:
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
        return vocab,word_to_idx,idx_to_word,
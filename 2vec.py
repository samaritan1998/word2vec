import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud #这个没有学过
from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA=torch.cuda.is_available()
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

#设置超参数
C=3
K=100
NUM_EPOCHS=2
MAX_VOCAB_SIZE=30000
BATCH_SIZE=128
LEARNING_RATE=0.2
EMBEDDING_SIZE=100


def word_tokenize(text):
    return text.split()

with open("text8.train.txt","r") as fin:
    text=fin.read()
text=text.split()
vocab=dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<unk>"]=len(text)-np.sum(list(vocab.values()))
idx_to_word=[word for word in vocab.keys()]
word_to_idx={word:i for i,word in enumerate(idx_to_word)}

#需要知道单词频率，一会儿负采样

word_counts=np.array([count for count in vocab.values()])
word_freqs=word_counts/np.sum(word_counts)
word_freqs=word_freqs**(3./4.)

VOCAB_SIZE=len(idx_to_word)

#实现DataLoader
'''
class WordEmbeddingDataset(tud.dataset):
    def __init__(self,text,word_to idx,idx_to_word,word_freqs,word_counts):
        super(WordEmbeddingDataset,self).__init__()
        self.text_encoded=[word_to_idx.get(word,word_to_idx["<unk>"]) for word in text]
        self.text_encoded=torch.LongTensor(self.text_encoded)
        self.word_to_idx=word_to_idx
        self.idx_to_word=idx_to_word
        self.word_counts=word_counts
    def __len__(self):
    def __getitem__(self,idx)
'''















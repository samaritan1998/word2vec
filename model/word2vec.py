import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math
import config

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from data.dataProcess import WordEmbeddingDataset,dataProcess
opt=config.DefaultConfig()
dataProcess=dataProcess()
BATCH_SIZE = opt.BATCH_SIZE
text=dataProcess.getText()
word_to_idx=dataProcess.getWordidx()
idx_to_word=dataProcess.getIdxword()
VOCAB_SIZE=dataProcess.getVocabsize()
word_counts=dataProcess.getWordcounts()
word_freqs=dataProcess.getWordfreq()
print("main里面的VOCAB_SIZE:",VOCAB_SIZE)
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts,VOCAB_SIZE)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):  # 参数跟在self里面，init里装了所有需要的模型，forward来安排顺序

        ''' 初始化输出和输出embedding
        '''
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.out_embed.weight.data.uniform_(-initrange, initrange)
        # 上面是初始的embedding长什么样子
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        self.in_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]

        return: loss, [batch_size]
        '''

        batch_size = input_labels.size(0)

        input_embedding = self.in_embed(input_labels)  # B * embed_size
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size

        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()









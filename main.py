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
from eval import evaluate,find_nearest
import re
from model import word2vec
from data.dataProcess import WordEmbeddingDataset,dataProcess

opt=DefaultConfig()
MAX_VOCAB_SIZE=opt.MAX_VOCAB_SIZE
EMBEDDING_SIZE=opt.EMBEDDING_SIZE
USE_CUDA=opt.USE_CUDA
NUM_EPOCHS=opt.NUM_EPOCHS
LOG_FILE=opt.LOG_FILE

LEARNING_RATE=opt.LEARNING_RATE
BATCH_SIZE=opt.BATCH_SIZE

dataProcess=dataProcess()
#WordEmbeddingDataset=WordEmbeddingDataset()
#word2vec=word2vec()

text=dataProcess.getText()
word_to_idx=dataProcess.getWordidx()
idx_to_word=dataProcess.getIdxword()
VOCAB_SIZE=dataProcess.getVocabsize()
word_counts=dataProcess.getWordcounts()
word_freqs=dataProcess.getWordfreq()

model=word2vec.EmbeddingModel(VOCAB_SIZE,EMBEDDING_SIZE)
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts,VOCAB_SIZE)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

#model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()



optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        # TODO
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()
        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            with open(LOG_FILE, "a") as fout:
                fout.write("epoch: {}, iter: {}, loss: {}\n".format(e, i, loss.item()))
                print("epoch: {}, iter: {}, loss: {}".format(e, i, loss.item()))

        if i % 2000 == 0:
            embedding_weights = model.input_embeddings()
            sim_simlex = evaluate("./data/simlex-999.txt", embedding_weights)
            sim_men = evaluate("./data/men.txt", embedding_weights)
            sim_353 = evaluate("./data/wordsim353.csv", embedding_weights)
            with open(LOG_FILE, "a") as fout:
                print("epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                    e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))
                fout.write(
                    "epoch: {}, iteration: {}, simlex-999: {}, men: {}, sim353: {}, nearest to monster: {}\n".format(
                        e, i, sim_simlex, sim_men, sim_353, find_nearest("monster")))

    embedding_weights = model.input_embeddings()
    np.save("embedding-{}".format(EMBEDDING_SIZE), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))

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
from data.dataProcess import WordEmbeddingDataset,dataProcess
import config
from config import DefaultConfig
from model import word2vec
opt=DefaultConfig()
dataProcess=dataProcess()
MAX_VOCAB_SIZE=opt.MAX_VOCAB_SIZE
EMBEDDING_SIZE=opt.EMBEDDING_SIZE
USE_CUDA=opt.USE_CUDA
NUM_EPOCHS=opt.NUM_EPOCHS
LOG_FILE=opt.LOG_FILE

LEARNING_RATE=opt.LEARNING_RATE
BATCH_SIZE=opt.BATCH_SIZE
text=dataProcess.getText()
word_to_idx=dataProcess.getWordidx()
idx_to_word=dataProcess.getIdxword()
VOCAB_SIZE=dataProcess.getVocabsize()
word_counts=dataProcess.getWordcounts()
word_freqs=dataProcess.getWordfreq()
model=word2vec.EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts,VOCAB_SIZE)
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

if USE_CUDA:
    model = model.cuda()
def evaluate(filename, embedding_weights):
    if filename.endswith(".csv"):
        data = pd.read_csv(filename, sep=",")
    else:  #只需要txt的这个就可以了
        data = pd.read_csv(filename, sep="\t")
    human_similarity = []
    model_similarity = []
    for i in data.iloc[:, 0:2].index:
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        if word1 not in word_to_idx or word2 not in word_to_idx:
            #model_similarity.append("OOV")
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            model_similarity.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
            human_similarity.append(float(data.iloc[i, 2]))
            #print(model_similarity)
            #print("----------------------------------------------")
            #print(human_similarity)
            #print("----------------------------------------------")
    return scipy.stats.spearmanr(human_similarity, model_similarity)# , model_similarity

def find_nearest(word):
    index = word_to_idx[word]
    embedding_weights = model.input_embeddings()
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:10]]

def caculate(evalFile,embedding_weights):
    out=[]
    data = pd.read_csv(evalFile, sep="\t",error_bad_lines=False)
    for i in data.iloc[:, 0:2].index:
        temp = []
        word1, word2 = data.iloc[i, 0], data.iloc[i, 1]
        temp.append(word1)
        temp.append(word2)
        if word1 not in word_to_idx or word2 not in word_to_idx:
            #model_similarity.append("OOV")
            temp.append("OOV")
            continue
        else:
            word1_idx, word2_idx = word_to_idx[word1], word_to_idx[word2]
            word1_embed, word2_embed = embedding_weights[[word1_idx]], embedding_weights[[word2_idx]]
            temp.append(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))
        out.append(temp)
    df = pd.DataFrame(out)

    df.to_csv('result.txt', sep='\t', index=False,header=False,encoding="utf-8")

def caculate1(evalFile,embedding_weights):
    fr=open('./2019110764.txt', 'w')
    with open(evalFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line=line.strip()
            words=line.split('\t')
            print(words)
            word1=words[0]
            word2=words[1]
            if word1 not in word_to_idx or word2 not in word_to_idx:
                fr.write(word1+'\t'+word2+'\t'+'OOV'+'\n')
            else:
                word1_idx,word2_idx = word_to_idx[word1],word_to_idx[word2]
                word1_embed,word2_embed=embedding_weights[[word1_idx]],embedding_weights[[word2_idx]]
                fr.write(word1 + '\t' + word2 + '\t' + str(float(sklearn.metrics.pairwise.cosine_similarity(word1_embed, word2_embed)))+'\n')
    fr.close()
    f.close()
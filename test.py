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
from eval import evaluate,find_nearest,caculate
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

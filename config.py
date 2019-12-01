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


class DefaultConfig(object):
    def __init__(self):
        self.path_data="./data"  #数据存放路径
        self.USE_CUDA=torch.cuda.is_available()
        self.K=100  #负采样的单词个数
        self.C=2 #窗口大小
        self.MAX_VOCAB_SIZE = 30000
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.2
        self.EMBEDDING_SIZE=100   #100维
        self.LOG_FILE = "word-embedding.log"
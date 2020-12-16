# 统一导入工具包
import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from torchtext.data.utils import get_tokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


if __name__=="__main__":
    # 讽刺
    '''
    torchtext.data.Field 声明处理数据的方式
    参数说明：
        tokenize 分词处理
        init_token 定义开始字符
        eos_token 定义结束字符
        lower 小写化处理
    '''
    # 声明处理方式，主要包括分词和小写化处理
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root='datas', train='wiki.train.tokens', validation='wiki.valid.tokens', test='wiki.test.tokens')

    print(f'train_txt: {len(train_txt.examples[0].text)}')



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
from model import TransformerModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                    init_token='<sos>',
                                    eos_token='<eos>',
                                    lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT, root='datas', train='wiki.train.tokens',
                                                                       validation='wiki.valid.tokens',
                                                                       test='wiki.test.tokens')

    # 依据训练集构建词典
    TEXT.build_vocab(train_txt)

    model = TransformerModel(len(TEXT.vocab.stoi), ninp=200, nhead=2, nhid=200, nlayers=2, dropout=0.2).to(device)
    # 模型加载训练好的参数
    # checkpoint = torch.load('datasets/models/best_model.pth.tar')
    checkpoint = torch.load('temp/models/best_model.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    # 已知序列
    history = 'it seems'
    h = []
    for w in history.split():
        h.append([TEXT.vocab.stoi[w]])

    while (True):
        # 把列表转化成 tensor ，然后计算模型输出
        output = model(torch.tensor(h).to(device))
        # 获取概率最大的5个单词的 id
        idxs = output[-1].argsort(descending=True).view(-1)[:10]
        # 随机选择其中一个
        r = random.randint(0, 10)
        h.append([r])
        # 句子结束
        if TEXT.vocab.itos[r] == '.' or TEXT.vocab.itos[r] == '<eos>':
            break

    # 将下标转化成句子
    sent = ''
    for w in h:
        sent += TEXT.vocab.itos[w[0]] + ' '

    # out_path = './tmp/hypotheses.txt'
    out_path = './temp/hypotheses.txt'
    # out_path = './submit/hypotheses.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('history: ' + history + '\n')
        f.write('hypotheses: ' + sent + '\n')

    print(sent)
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


class PositionalEncoding(nn.Module):
    '''
    给原始序列添加位置编码
    '''
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 首先初始化为0
        pe = torch.zeros(max_len, d_model)
        # unsqueeze()，输入(1,3)，在第二维增加一个维度，使其维度变为（2，1，3）
        # squeeze()，输入(2,1,3)，在第二维去掉一个维度，变为(1,3)
        # arange(0,n), 从0到n，不包含n；range(0,n),从0到n，包含n
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # sine 和 cosine 来生成位置信息
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 词经过嵌入层后，再加上位置信息
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    '''
    ntoken: 词表大小，用于构建嵌入层
    ninp: 模型维度
    nhead: 多头注意力机制中 head 数目
    nhid: 前馈神经网络的维度
    nlayers: TransformerEncoderLayer叠加层数
    '''

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)  # 位置编码
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)  # EncoderLayer
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)  # Encoder
        self.encoder = nn.Embedding(ntoken, ninp)  # 嵌入层
        self.ninp = ninp  # 模型维度

        # decoder 用于将隐藏层的表示转化成词表中 token 的概率分布
        self.decoder = nn.Linear(ninp, ntoken)

    def forward(self, src):
        # 生成 mask ，保证模型只能看到当前位置之前的信息
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)  # 位置编码
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=None)
        output = self.decoder(output)
        return output


def subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


if __name__ == "__main__":
    plt.figure(figsize=(5, 5))
    plt.imshow(subsequent_mask(20))
    plt.show()
    print('pause')
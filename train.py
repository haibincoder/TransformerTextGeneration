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

# 获取当前设备
from model import TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    '''
    将数据划分为用于训练的批次
    '''
    # 将文本形式的数据用 token 的相应索引来表示
    data = TEXT.numericalize([data.examples[0].text])
    # 获取总的批次数目
    nbatch = data.size(0) // bsz
    # 去除剩余的部分，比如总长度为12，而批次大小为5，那么剩余的2个 token 将不会被包括在内
    # narrow()获取指定维度的值
    data = data.narrow(0, 0, nbatch * bsz)
    # 根据批次大小，划分数据集
    # view(-1,x) -1为自动计算大小，x为指定维度，例如维度[20,1].view(5, -1)，结果维度为[5,4]
    # t()转置矩阵，因为view()是引用之前内存，需要contiguous()复制到新的内存，并根据转置矩阵排序
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


bptt = 35  # 句子长度
epoch = 10

def get_batch(source, i):
        '''
        把数据进一步切分成长度为35的序列，最后返回的 data:[35, batch_size] ,每一列表示一个连续的序列
        '''
        seq_len = min(bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].view(-1)
        return data, target


def train():
    model.train()  # 训练模式，更新模型参数
    total_loss = 0.
    start_time = time.time()  # 用于记录模型的训练时长
    ntokens = len(TEXT.vocab.stoi)  # 词表大小
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 获取批次数据
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        # 计算损失
        loss = criterion(output.view(-1, ntokens), targets)
        # 计算梯度
        loss.backward()
        # 梯度裁剪，防止梯度消失/爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # 优化参数
        optimizer.step()

        # 打印训练记录
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    eval_model.eval() # 评估模式，不更新模型参数，仅评估模型当前的表现
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi) # 词表大小
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == "__main__":
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

    print(f'train_txt tokens: {len(train_txt.examples[0].text)}')
    print(f'val_txt tokens: {len(val_txt.examples[0].text)}')
    print(f'test_txt tokens: {len(test_txt.examples[0].text)}')

    # 依据训练集构建词典
    TEXT.build_vocab(train_txt)

    # 查看词典
    length = len(TEXT.vocab)
    print('词表大小: %d' % length)
    print(TEXT.vocab.stoi)
    print(f'"the": {TEXT.vocab["the"]}')

    batch_size = 15
    eval_batch_size = 10
    train_data = batchify(train_txt, batch_size)
    val_data = batchify(val_txt, eval_batch_size)
    test_data = batchify(test_txt, eval_batch_size)

    # 数据处理，torch.nn.TransformerEncoder 要求的输入格式为 src:(S, N, E)，其中 S 表示句子长度，N 表示批次大小，E 表示模型维度
    data, targets = get_batch(train_data, 0)
    print(data.size())
    print(targets.size())  # target 表示待预测的下一个正确的词，用于计算模型损失，进而更新参数

    # 打印测试数据
    src = ''
    for id in data[:, 0]:
        src = src + (' %s' % TEXT.vocab.itos[id])
    src.strip()
    tgt = ''
    for id in targets[0::20]:
        tgt = tgt + (' %s' % TEXT.vocab.itos[id])
    tgt.strip()

    # transformer参数
    criterion = nn.CrossEntropyLoss()
    ntokens = len(TEXT.vocab.stoi)  # 词表大小
    emsize = 200  # 嵌入层维度
    nhid = 200  # nn.TransformerEncoder 中前馈神经网络的维度
    nlayers = 2  # 编码器中 nn.TransformerEncoderLayer 层数
    nhead = 2  # 多头注意力机制中“头”的数目
    dropout = 0.2  # dropout
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    # xavier_normal_初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    # 学习率
    lr = 2.0
    # 随机梯度下降
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # 动态调整学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    best_val_loss = float("inf")
    epochs = 3  # 共训练3个epoch
    best_model = None

    for epoch in range(1, epochs + 1):
        # for epoch in range(1, 2): # Kagging test
        epoch_start_time = time.time()
        # 训练过程
        train()
        # 验证过程
        val_loss = evaluate(model, val_data)
        # 打印验证结果
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        # 记录最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        # 调整学习率
        scheduler.step()

    # # 保存模型
    # if not os.path.exists('datasets/models'):
    #     os.makedirs('datasets/models')
    # torch.save({'state_dict': model.state_dict()}, 'datasets/models/best_model.pth.tar')

    # 保存模型
    if not os.path.exists('temp/models'):
        os.makedirs('temp/models')
    torch.save({'state_dict': model.state_dict()}, 'temp/models/best_model.pth.tar')
    print('train finish')

    # test
    # 计算交叉熵损失
    test_loss = evaluate(best_model, test_data)

    # 计算困惑度
    ppl = math.exp(test_loss)
    print('=' * 40)
    print('| End of training | test ppl {:8.2f}'.format(ppl))
    print('=' * 40)






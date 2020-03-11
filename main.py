import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, OneCycleLR
from torch.utils.data import Dataset, Subset, DataLoader

import sys
import math
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

import pytorch_lightning as pl

import text8dataset
import lmmodels


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = 10000
    seq_len = 20
    with open(sys.argv[1]) as f:
        text = f.read()
    tr_text_len = int(0.95 * len(text))
    va_text_len = int(0.04 * len(text))
    te_text_len = int(0.01 * len(text))
    tr_ds = text8dataset.Text8WordDataSet(text[:tr_text_len], seq_len=seq_len, max_vocab_size=vocab_size)
    va_ds = text8dataset.Text8WordDataSet(text[tr_text_len:tr_text_len+va_text_len], seq_len=seq_len, vocab=tr_ds.vocab)
    te_ds = text8dataset.Text8WordDataSet(text[tr_text_len+va_text_len:], seq_len=seq_len, vocab=tr_ds.vocab)
    ds_len = len(tr_ds)
    print(ds_len, seq_len, vocab_size)
    bs = 128
    va_bs = bs
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=va_bs)
    te_dl = DataLoader(te_ds, batch_size=va_bs)
    print(len(tr_dl))

    hidden_size = 1024
    emb_size = 128
    num_layers = 1
    model = lmmodels.RNNLanguageModel(
            vocab_size, emb_size, hidden_size, num_layers,
            'lstm', tr_dl, va_dl, te_dl)
    trainer = pl.Trainer()
    epochs = 25

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
    vocab_size = 10000
    seq_len = 20
    with open(sys.argv[1]) as f:
        text = f.read()
    # text = text[:20000]
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
    model = lmmodels.SimpleLanguageModel(
            vocab_size, emb_size, hidden_size, num_layers,
            'lstm', tr_dl, va_dl, te_dl)
    epochs = 25
    logger = pl.loggers.TensorBoardLogger('./output/', name='language_model')
    class SentenceGenerationCallback(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            try:
                print('Generated sentence: {}\n'.format(' '.join([tr_ds.index_to_word[i] for i in pl_module.generate_sentence(seq_len * 2)])))
            except Exception as e:
                print('Sentence generation failed with {}'.format(e))
                raise e
    trainer = pl.Trainer(
            fast_dev_run=False,
            gradient_clip_val=0.25,
            max_epochs=epochs,
            default_save_path='./output/',
            logger=logger,
            callbacks=[SentenceGenerationCallback()],
            check_val_every_n_epoch=1)
    trainer.fit(model)

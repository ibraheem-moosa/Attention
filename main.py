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

from ignite.engine import Events, Engine, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import text8dataset
import lmmodels


class CrossEntropyLanguageModel(nn.Module):
    def __init__(self):
        super(CrossEntropyLanguageModel, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inp, targ):
        inp = inp.view((-1, inp.shape[-1]))
        targ = targ.view((-1,))
        return self.ce(inp, targ)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = text8dataset.Text8WordDataSet(sys.argv[1], seq_len=20, max_vocab_size=10000)
    ds_len = len(ds)
    print(ds_len, ds.vocab_size)
    indices = list(range(ds_len))
    random.seed(42)
    random.shuffle(indices)
    te_ratio = 0.01
    te_ds_len = int(ds_len * te_ratio)
    va_ratio = 0.04
    va_ds_len = int(ds_len * va_ratio)
    tr_indices = indices[:-va_ds_len-te_ds_len]
    va_indices = indices[-va_ds_len-te_ds_len:-te_ds_len]
    te_indices = indices[-te_ds_len:]
    tr_ds, va_ds, te_ds = Subset(ds, tr_indices), Subset(ds, va_indices), Subset(ds, te_indices)
    bs = 2048
    va_bs = bs
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=va_bs)
    te_dl = DataLoader(te_ds, batch_size=va_bs)
    print(len(tr_dl))

    hidden_size = 512
    emb_size = 128
    num_layers = 2
    model = lmmodels.SimpleGRULanguageModel(ds.vocab_size, emb_size, hidden_size, num_layers).to(device)
    optimizer = Adam(model.parameters(), lr=1e2)
    criterion = CrossEntropyLanguageModel()
    lr_finder_baselr = 1e-4
    lr_finder_maxlr = 1e0
    lr_finder_steps = 100
    lr_finder_gamma = (lr_finder_maxlr / lr_finder_baselr) ** (1 / lr_finder_steps)
    lr_finder_scheduler = LambdaLR(optimizer,
            lambda e: lr_finder_baselr * (lr_finder_gamma ** e))

    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    lr_finder = Engine(update_model)
    metrics = {
            'acc': Accuracy(
                output_transform=lambda y_pred: (y_pred[0].view((-1, ds.vocab_size)), y_pred[1].view((-1,)))),
            'ce': Loss(criterion)}
    lr_finder_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    lr_finder_va_ce = []
    lr_finder_va_acc = []
    @lr_finder.on(Events.ITERATION_COMPLETED)
    def step_lr_finder_sched(lr_finder):
        lr_finder_scheduler.step()
        lr_finder_evaluator.run(va_dl)
        metrics = lr_finder_evaluator.state.metrics
        lr_finder_va_ce.append(metrics['ce'])
        lr_finder_va_acc.append(metrics['acc'])
        if math.isnan(metrics['ce']) or math.isnan(lr_finder.state.output):
            lr_finder.fire_event(Events.COMPLETED)
    @lr_finder.on(Events.COMPLETED)
    def set_lr(lr_finder):
        plt.plot(np.minimum(lr_finder_va_ce, 9.3))
        plt.show()
        plt.plot(lr_finder_va_acc)
        plt.show()
        sys.exit()

    # lr_finder.run(tr_dl, epoch_length=lr_finder_steps)

    epochs = 25
    # scheduler = OneCycleLR(optimizer, max_lr=1e-2, epochs=epochs, steps_per_epoch=len(tr_dl), pct_start=0.5, anneal_strategy='linear')
    scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    trainer = Engine(update_model)
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    """
    @trainer.on(Events.ITERATION_COMPLETED)
    def scheduler_step(trainer):
        scheduler.step()
    """
    @trainer.on(Events.ITERATION_COMPLETED(every=16))
    def log_tr_loss(trainer):
        print(datetime.datetime.now())
        print('Epoch {} Iter: {}: Loss: {:.6f}'.format(trainer.state.epoch, trainer.state.iteration, trainer.state.output))
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_va_loss(trainer):
        evaluator.run(va_dl)
        metrics = evaluator.state.metrics
        print('Epoch {}: Va Acc: {:.6f} Va Loss: {:.6f}'.format(trainer.state.epoch, metrics['acc'], metrics['ce']))
        scheduler.step(metrics['ce'])
    @trainer.on(Events.COMPLETED)
    def log_tr_loss(trainer):
        va_ce = evaluator.state.metrics['ce']
        evaluator.run(tr_dl)
        metrics = evaluator.state.metrics
        tr_ce = metrics['ce']
        generalization_error = va_ce - tr_ce
        print('Epoch {}: Tr Acc: {:.6f} Tr Loss: {:.6f} Ge Error: {:.6f}'.format(trainer.state.epoch, metrics['acc'], metrics['ce'], generalization_error))

    trainer.run(tr_dl, max_epochs=epochs)

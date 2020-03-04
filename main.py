import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, DataLoader

import random
import datetime
from collections import Counter, defaultdict

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
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
    ds = text8dataset.Text8CharDataSet('./text8', seq_len=100)
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
    bs = 128
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=bs)
    te_dl = DataLoader(te_ds, batch_size=bs)
    print(len(tr_dl))

    hidden_size = 128
    num_layers = 1
    model = lmmodels.SimpleRNNLanguageModel(ds.vocab_size, hidden_size, num_layers)
    optimizer = Adam(model.parameters())
    loss = CrossEntropyLanguageModel()
    scheduler = ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    metrics = {
            'acc': Accuracy(
                output_transform=lambda y_pred: (y_pred[0].view((-1, ds.vocab_size)), y_pred[1].view((-1,)))),
            'ce': Loss(loss)}
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=500))
    def log_tr_loss(trainer):
        print(datetime.datetime.now())
        print('Epoch {} Iter: {}: Loss: {:.6f}'.format(trainer.state.epoch, trainer.state.iteration, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_va_loss(trainer):
        evaluator.run(tr_dl)
        metrics = evaluator.state.metrics
        print('Epoch {}: Tr Acc: {:.6f} Tr Loss: {:.6f}'.format(trainer.state.epoch, metrics['acc'], metrics['ce']))
        evaluator.run(va_dl)
        metrics = evaluator.state.metrics
        print('Epoch {}: Va Acc: {:.6f} Va Loss: {:.6f}'.format(trainer.state.epoch, metrics['acc'], metrics['ce']))
        scheduler.step(metrics['ce'])

    trainer.run(tr_dl, max_epochs=25)

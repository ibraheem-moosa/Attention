import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, Subset, DataLoader

import random
import datetime
from collections import Counter, defaultdict

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss

import text8dataset


class SimpleRNNLanguageModel(nn.Module):
    "A simple one directional RNN that predicts next character."

    def __init__(self, vocab_size, hidden_size, num_layers):
        super(SimpleRNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size) 
        self.rnn = nn.RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        self.final = nn.Linear(hidden_size, vocab_size, bias=False)

    "Input shape: (bs, seq len)"
    def forward(self, x):
        x = self.embedding(x)
        x, h_n = self.rnn(x)
        x = self.final(F.relu(x))
        return x


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
    ds = text8dataset.Text8WordDataSet('./text8', 10, max_vocab_size=10000)
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
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=bs)
    te_dl = DataLoader(te_ds, batch_size=bs)
    print(len(tr_dl))

    hidden_size = 8
    num_layers = 1
    model = SimpleRNNLanguageModel(ds.vocab_size, hidden_size, num_layers)
    optimizer = SGD(model.parameters(), lr=1e-1, nesterov=True, momentum=0.9)
    loss = CrossEntropyLanguageModel()

    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    metrics = {
            'acc': Accuracy(
                output_transform=lambda y_pred: (y_pred[0].view((-1, ds.vocab_size)), y_pred[1].view((-1,)))),
            'ce': Loss(loss)}
    evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

    @trainer.on(Events.ITERATION_COMPLETED(every=10))
    def log_tr_loss(trainer):
        print(datetime.datetime.now())
        print('Epoch {} Iter: {}: Loss: {:.6f}'.format(trainer.state.epoch, trainer.state.iteration, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_va_loss(trainer):
        evaluator.run(va_dl)
        metrics = evaluator.state.metrics
        print('Epoch {}: Va Acc: {:.6f} Va Loss: {:.6f}'.format(trainer.state.epoch, metrics['acc'], metrics['ce']))

    trainer.run(tr_dl, max_epochs=10)

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, Subset, DataLoader

import random
import datetime

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


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

class Text8DataSet(Dataset):
    def __init__(self, path, seq_len):
        super(Text8DataSet, self).__init__()
        self.seq_len = seq_len
        with open(path) as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        assert(self.vocab_size < 255)
        index_to_char = dict((i, c) for (i, c) in enumerate(chars))
        char_to_index = dict((c, i) for (i, c) in enumerate(chars))
        self.index_to_char = index_to_char
        self.char_to_index = char_to_index
        text = [self.char_to_index[c] for c in text]
        text = torch.tensor(text, dtype=torch.uint8)
        self.text = text
        self.length = (text.shape[0] - 1) // seq_len # reserve one character at end

    def __getitem__(self, idx):
        x = self.text[idx * self.seq_len: (idx + 1) * self.seq_len]
        y = self.text[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]
        return x.to(torch.long), y.to(torch.long)

    def __len__(self):
        return self.length

class CrossEntropyLanguageModel(nn.Module):
    def __init__(self):
        super(CrossEntropyLanguageModel, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inp, targ):
        inp = inp.view((-1, inp.shape[-1]))
        targ = targ.view((-1,))
        return self.ce(inp, targ)

if __name__ == '__main__':
    ds = Text8DataSet('./text8', 10)
    ds_len = len(ds)
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
    print(len(tr_dl))

    hidden_size = 8
    num_layers = 1
    model = SimpleRNNLanguageModel(ds.vocab_size, hidden_size, num_layers)
    optimizer = SGD(model.parameters(), lr=1e-3, nesterov=True, momentum=0.9)
    loss = CrossEntropyLanguageModel()

    trainer = create_supervised_trainer(model, optimizer, loss)
    evaluator = create_supervised_evaluator(model, metrics={'acc': Accuracy(), 'ce': Loss(loss)})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_tr_loss(trainer):
        if trainer.state.iteration % 100 == 0:
            print(datetime.datetime.now())
            print('Epoch {} Iter: {}: Loss: {:.6f}'.format(trainer.state.epoch, trainer.state.iteration, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_va_loss(trainer):
        evaluator.run(va_dl)
        metrics = evaluator.state.metrics
        print('Epoch {}: Va Acc: {:.6f} Va Loss: {:.6f}'.format(trainer.state.epoch, metrics['acc'], metrics['ce']))

    trainer.run(tr_dl, max_epochs=10)

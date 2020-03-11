"Language Models"
import torch
from torch import nn
import torch.nn.functional as F
import pythorch_lightning as pl

class CrossEntropyLanguageModel(nn.Module):
    def __init__(self):
        super(CrossEntropyLanguageModel, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inp, targ):
        inp = inp.view((-1, inp.shape[-1]))
        targ = targ.view((-1,))
        return self.ce(inp, targ)


class SimpleLanguageModel(pl.LightningModule):
    "A simple one directional RNN that predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, rnn_type):
        super(SimpleRNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, max_norm=1.0)
        self.rnn_type = rnn_type
        if rnn_type == 'rnn-relu':
            self.rnn = nn.RNN(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='relu',
                bias=False,
                batch_first=True)
        elif rnn_type == 'rnn-tanh':
            self.rnn = nn.RNN(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                bias=False,
                batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        else:
            raise ValueError('rnn_type {} not supported'.format(rnn_type))
        self.projection = nn.Linear(hidden_size, emb_size, bias=False)
        self.out_emb = nn.Linear(emb_size, vocab_size, bias=False)
        self.initialize()

    def initialize(self):
        if self.rnn_type == 'rnn-relu':
            for name, param in self.rnn.named_parameters():
                if name.startswith('weight_hh'):
                    nn.init.eye_(param)
                if name.startswith('weight_ih'):
                    nn.init.kaiming_normal_(param)
            nn.init.kaiming_normal_(self.projection.weight)
            nn.init.kaiming_normal_(self.out_emb.weight)

    "Input shape: (bs, seq len)"
    def forward(self, x):
        x = self.embedding(x)
        x, h_n = self.rnn(x)
        x = self.projection(F.relu(x))
        x = self.out_emb(x)
        return x


class SharedEmbeddingLanguageModel(SimpleRNNLanguageModel):
    "A one directional RNN that shares input and output embedding and predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(RNNSharedEmbeddingLanguageModel, self).__init__(vocab_size, emb_size, hidden_size, num_layers)
        self.out_emb.weight = self.embedding.weight

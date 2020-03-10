"Language Models"
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class SimpleRNNLanguageModel(nn.Module):
    "A simple one directional RNN that predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(SimpleRNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='relu',
                bias=False,
                batch_first=True)
        self.projection = nn.Linear(hidden_size, emb_size, bias=False)
        self.out_emb = nn.Linear(emb_size, vocab_size, bias=False)
        # do initalization
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

    def generate_sentence(self, length):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval()
        with torch.no_grad():
            sentence = []
            current_char = np.random.randint(self.embedding.num_embeddings)
            sentence.append(current_char)
            h = torch.zeros((self.rnn.num_layers, 1, self.rnn.hidden_size)).to(device)
            for i in range(length):
                x = torch.tensor(current_char, dtype=torch.long).to(device).reshape((1, 1))
                x = self.embedding(x)
                x, h = self.rnn(x, h)
                x = self.projection(F.relu(x))
                x = self.out_emb(x)
                x = x.view((self.embedding.num_embeddings,))
                x = F.softmax(x)
                x = x.cpu()
                x = x / x.sum()
                current_char = np.random.multinomial(1, x).nonzero()[0].item()
                sentence.append(current_char)
        return sentence

class RNNSharedEmbeddingLanguageModel(SimpleRNNLanguageModel):
    "A one directional RNN that shares input and output embedding and predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(RNNSharedEmbeddingLanguageModel, self).__init__(vocab_size, emb_size, hidden_size, num_layers)
        self.out_emb.weight = self.embedding.weight


class SimpleLSTMLanguageModel(nn.Module):
    "A simple one directional LSTM RNN that predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(SimpleLSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        self.projection = nn.Linear(hidden_size, emb_size, bias=False)
        self.out_emb = nn.Linear(emb_size, vocab_size, bias=False)
        # do initalization
        """
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_hh'):
                nn.init.eye_(param)
            if name.startswith('weight_ih'):
                nn.init.kaiming_normal_(param)
        nn.init.kaiming_normal_(self.out_emb.weight)
        """

    "Input shape: (bs, seq len)"
    def forward(self, x):
        x = self.embedding(x)
        x, (h_n, c_n) = self.rnn(x)
        x = self.projection(F.relu(x))
        x = self.out_emb(x)
        return x

    def generate_sentence(self, length):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval()
        with torch.no_grad():
            sentence = []
            current_char = np.random.randint(self.embedding.num_embeddings)
            sentence.append(current_char)
            h = torch.zeros((self.rnn.num_layers, 1, self.rnn.hidden_size)).to(device)
            for i in range(length):
                x = torch.tensor(current_char, dtype=torch.long).to(device).reshape((1, 1))
                x = self.embedding(x)
                x, h = self.rnn(x, h)
                x = self.projection(F.relu(x))
                x = self.out_emb(x)
                x = x.view((self.embedding.num_embeddings,))
                x = F.softmax(x)
                x = x.cpu()
                x = x / x.sum()
                current_char = np.random.multinomial(1, x).nonzero()[0].item()
                sentence.append(current_char)
        return sentence



class LSTMSharedEmbeddingLanguageModel(SimpleLSTMLanguageModel):
    "A one directional LSTM RNN that shares input and output embedding and predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(LSTMSharedEmbeddingLanguageModel, self).__init__(vocab_size, emb_size, hidden_size, num_layers)
        self.out_emb.weight = self.embedding.weight


class SimpleGRULanguageModel(nn.Module):
    "A simple one directional GRU RNN that predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(SimpleGRULanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.GRU(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        self.projection = nn.Linear(hidden_size, emb_size, bias=False)
        self.out_emb = nn.Linear(emb_size, vocab_size, bias=False)
        # do initalization
        """
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight_hh'):
                nn.init.eye_(param)
            if name.startswith('weight_ih'):
                nn.init.kaiming_normal_(param)
        nn.init.kaiming_normal_(self.out_emb.weight)
        """

    "Input shape: (bs, seq len)"
    def forward(self, x):
        x = self.embedding(x)
        x, h_n = self.rnn(x)
        x = self.projection(F.relu(x))
        x = self.out_emb(x)
        return x

    def generate_sentence(self, length):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval()
        with torch.no_grad():
            sentence = []
            current_char = np.random.randint(self.embedding.num_embeddings)
            sentence.append(current_char)
            h = torch.zeros((self.rnn.num_layers, 1, self.rnn.hidden_size)).to(device)
            for i in range(length):
                x = torch.tensor(current_char, dtype=torch.long).to(device).reshape((1, 1))
                x = self.embedding(x)
                x, h = self.rnn(x, h)
                x = self.projection(F.relu(x))
                x = self.out_emb(x)
                x = x.view((self.embedding.num_embeddings,))
                x = F.softmax(x)
                x = x.cpu()
                x = x / x.sum()
                current_char = np.random.multinomial(1, x).nonzero()[0].item()
                sentence.append(current_char)
        return sentence


class GRUSharedEmbeddingLanguageModel(SimpleGRULanguageModel):
    "A one directional GRU RNN that shares input and output embedding and predicts next character."

    def __init__(self, vocab_size, emb_size, hidden_size, num_layers):
        super(GRUSharedEmbeddingLanguageModel, self).__init__(vocab_size, emb_size, hidden_size, num_layers)
        self.out_emb.weight = self.embedding.weight

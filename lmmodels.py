"Language Models"
import torch
from torch import nn
import torch.nn.functional as F

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

class RNNSharedEmbeddingLanguageModel(SimpleRNNLanguageModel):
    "A one directional RNN that shares input and output embedding and predicts next character."

    def __init__(self, vocab_size, hidden_size, num_layers):
        super(RNNSharedEmbeddingLanguageModel, self).__init__(vocab_size, hidden_size, num_layers)
        self.final.weight = self.embedding.weight



import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

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
        assert(len(chars) < 255)
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
        return x, y

    def __len__(self):
        return self.length


if __name__ == '__main__':
    model = SimpleRNNLanguageModel(27, 8, 1)
    print(model(torch.zeros((10, 8), dtype=torch.long)).shape)
    ds = Text8DataSet('./text8', 10)
    print(len(ds))

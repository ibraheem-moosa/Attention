"Text8 Dataset"
import torch
from torch.utils.data import Dataset
from collections import Counter, defaultdict


class Text8CharDataSet(Dataset):
    def __init__(self, path, seq_len):
        super(Text8CharDataSet, self).__init__()
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

class Text8WordDataSet(Dataset):
    def __init__(self, path, seq_len=10, max_vocab_size=None):
        super(Text8WordDataSet, self).__init__()
        self.seq_len = seq_len
        with open(path) as f:
            text = f.read()
        text = text.split()
        word_counter = Counter(text)
        if max_vocab_size is None:
            max_vocab_size = len(word_counter)
        words = sorted(list(map(lambda t:t[0], word_counter.most_common(max_vocab_size))))
        self.vocab_size = len(words)
        assert self.vocab_size <= max_vocab_size
        index_to_word = defaultdict(lambda :'unk')
        index_to_word.update(((i, w) for (i, w) in enumerate(words)))
        word_to_index = defaultdict(lambda :0)
        word_to_index.update(((w, i) for (i, w) in enumerate(words)))
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        text = [self.word_to_index[w] for w in text]
        text = torch.tensor(text, dtype=torch.uint8)
        self.text = text
        self.length = (text.shape[0] - 1) // seq_len # reserve one character at end

    def __getitem__(self, idx):
        x = self.text[idx * self.seq_len: (idx + 1) * self.seq_len]
        y = self.text[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]
        return x.to(torch.long), y.to(torch.long)

    def __len__(self):
        return self.length

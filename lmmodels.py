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

def make_rnn(rnn_type, input_size, hidden_size, num_layers):
    if rnn_type == 'rnn-relu':
            return nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='relu',
                bias=False,
                batch_first=True)
        elif rnn_type == 'rnn-tanh':
            return nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity='tanh',
                bias=False,
                batch_first=True)
        elif rnn_type == 'gru':
            return nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        elif rnn_type == 'lstm':
            return nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=False,
                batch_first=True)
        else:
            raise ValueError('rnn_type {} not supported'.format(rnn_type))


class SimpleLanguageModel(pl.LightningModule):
    "A simple one directional RNN that predicts next character."

    def __init__(self, 
            vocab_size, 
            emb_size, 
            hidden_size, 
            num_layers, 
            rnn_type,
            tr_dl,
            va_dl,
            te_dl):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, max_norm=1.0)
        self.rnn_type = rnn_type
        self.rnn = make_rnn(rnn_type, emb_size, hidden_size, num_layers)
        self.projection = nn.Linear(hidden_size, emb_size, bias=False)
        self.out_emb = nn.Linear(emb_size, vocab_size, bias=False)
        self.initialize()
        self.criterion = CrossEntropyLanguageModel()
        self.prepare_data_callback = prepare_data_callback
        self.tr_dl = tr_dl
        self.va_dl = va_dl
        self.te_dl = te_dl

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

    def generate_sentence(self, length, start_with=None):
        self.eval()
        with torch.no_grad():
            sentence = []
            if start_with is None:
                start_with = [np.random.randint(self.embedding.num_embeddings)]
            sentence.append(start_with[0])
            h = torch.zeros((self.rnn.num_layers, 1, self.rnn.hidden_size)).type_as(self.embedding.weight.type())
            for current_token in start_with:
                x = torch.tensor(current_token, dtype=torch.long).type_as(self.embedding.weight.type()).reshape((1, 1))
                x = self.embedding(x)
                x, h = self.rnn(x, h)
                sentence.append(current_token)
            for i in range(length):
                x = torch.tensor(current_token, dtype=torch.long).type_as(self.embedding.weight.type()).reshape((1, 1))
                x = self.embedding(x)
                x, h = self.rnn(x, h)
                x = self.projection(F.relu(x))
                x = self.out_emb(x)
                x = x.view((self.embedding.num_embeddings,))
                x = F.softmax(x, dim=0)
                x = x.cpu().numpy().astype(np.float64)
                x /= x.sum()
                current_token = np.random.multinomial(1, x).nonzero()[0].item()
                sentence.append(current_token)
        return sentence

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)

        log = {'train_loss': loss}

        return {'loss': loss, 'log': log}

    def validation_step(self, validation_batch, batch_idx):
        x, y = validation_batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': log}

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return self.tr_dl

    def val_dataloader(self):
        return self.va_dl

    def test_dataloader(self):
        return self.te_dl

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class SharedEmbeddingLanguageModel(SimpleRNNLanguageModel):
    "A one directional RNN that shares input and output embedding and predicts next character."

    def __init__(self, 
            vocab_size, 
            emb_size, 
            hidden_size, 
            num_layers, 
            rnn_type,
            tr_dl,
            va_dl,
            te_dl):
        super(SharedEmbeddingLanguageModel, self).__init__(
                vocab_size, emb_size, hidden_size, num_layers,
                rnn_type, tr_dl, va_dl, te_dl)
        self.out_emb.weight = self.embedding.weight

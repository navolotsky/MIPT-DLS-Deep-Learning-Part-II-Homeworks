import random
import torch
from torch import nn
from torch.nn import functional as F


def softmax(x, temperature=10):  # use your temperature
    e_x = torch.exp(x / temperature)
    return e_x / torch.sum(e_x, dim=0)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional, padding_idx=0):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.internal_hid_dim = hid_dim
        self.output_hid_dim = hid_dim * (bidirectional + 1)
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            input_dim, emb_dim, padding_idx=padding_idx)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                           dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src):

        # src = [src sent len, batch size]

        # Compute an embedding from the src data and apply dropout to it
        embedded = self.dropout(self.embedding(src))

        # embedded = [src sent len, batch size, emb dim]

        # Compute the RNN output values of the encoder RNN.
        # outputs, hidden and cell should be initialized here. Refer to nn.LSTM docs ;)

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer
        if self.bidirectional:
            dec_init_hidden = (
                hidden
                .reshape(self.n_layers, 2, -1, self.internal_hid_dim)
                .transpose(1, 2)
                .reshape(self.n_layers, -1, self.output_hid_dim)
            )
            dec_init_cell = (
                cell
                .reshape(self.n_layers, 2, -1, self.internal_hid_dim)
                .transpose(1, 2)
                .reshape(self.n_layers, -1, self.output_hid_dim)
            )
        else:
            dec_init_hidden = hidden
            dec_init_cell = cell
        return outputs, dec_init_hidden, dec_init_cell


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, softmax_fn=None):
        super().__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim, 1)

        self.softmax = softmax_fn if softmax_fn is not None else softmax

    def forward(self, hidden, encoder_outputs):

        # encoder_outputs = [src sent len, batch size, enc_hid_dim]
        # hidden = [1, batch size, dec_hid_dim]

        # repeat hidden and concatenate it with encoder_outputs
        hidden_repeated = hidden.repeat_interleave(
            encoder_outputs.shape[0], dim=0)
        # calculate energy
        # print(f"{hidden_repeated.shape = }; {enc_outputs.shape = }")
        concatenated = torch.cat((hidden_repeated, encoder_outputs), dim=-1)
        energy = self.v(torch.tanh(self.attn(concatenated)))
        # get attention, use softmax function which is defined, can change temperature
        attn_weights = self.softmax(energy)

        return attn_weights


class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention, padding_idx=0):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.internal_hid_dim = self.output_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(
            output_dim, emb_dim, padding_idx=padding_idx)

        self.rnn = nn.GRU(emb_dim + enc_hid_dim, dec_hid_dim)

        # linear layer to get next word
        self.out = nn.Linear(emb_dim + enc_hid_dim + dec_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # because only one word, no words sequence

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        # get weighted sum of encoder_outputs
        attn_weights = self.attention(hidden, encoder_outputs)
        weighted = (attn_weights * encoder_outputs).sum(0, keepdim=True)
        # concatenate weighted sum and embedded, break through the GRU
        concatenated = torch.cat((embedded, weighted), dim=-1)
        outputs, _ = self.rnn(concatenated, hidden)
        # get predictions
        features = torch.cat((embedded, weighted, outputs), dim=-1).squeeze(0)

        # prediction = [batch size, output dim]
        preds = self.out(features)

        return preds


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        if not encoder.output_hid_dim == decoder.internal_hid_dim:
            raise ValueError(
                "Hidden dimensions of encoder and decoder must be equal!")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        # Again, now batch is the first dimension instead of zero
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_states, hidden, cell = self.encoder(src)

        # Select hidden state from only the last encoder layer.
        # It is required because the decoder has only one layer.
        hidden = hidden[-1:, ]
        # cell is not used.

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            output = self.decoder(input, hidden, enc_states)

            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            if teacher_force:
                input = trg[t]
            else:
                # get the highest predicted token from our predictions
                top1 = output.argmax(-1)
                input = top1

        return outputs


if __name__ == "__main__":
    input_dim = 1200  # words
    output_dim = 1400
    batch_size = 128
    emb_dim = 300
    enc_hid_dim = 50
    bidirectional = True
    enc_output_hid_dim = enc_hid_dim * (int(bidirectional) + 1)
    n_layers = 2
    dec_hid_dim = 100
    dropout = 0.3
    sent_length = 23

    batch = torch.randint(input_dim, (sent_length, batch_size))

    enc = Encoder(input_dim, emb_dim, enc_hid_dim,
                  n_layers, dropout, bidirectional)
    attn = Attention(enc_output_hid_dim, dec_hid_dim)
    dec = DecoderWithAttention(
        output_dim, emb_dim, enc_output_hid_dim, dec_hid_dim, dropout, attn)
    seq2seq = Seq2Seq(enc, dec)

    print(f"{[tensor.shape for tensor in enc(batch)]}")
    enc_outputs, dec_init_hidden, dec_init_cell = enc(batch)
    prev_word = batch[0]
    prev_hidden = dec_init_hidden[-1:, ]  # time = 0
    print(f"{attn(prev_hidden, enc_outputs).shape = }")

    print(f"{dec(prev_word, prev_hidden, enc_outputs).shape = }")

    src = torch.randint(input_dim, (sent_length, batch_size))
    trg = torch.randint(output_dim, (sent_length, batch_size))
    predicts = seq2seq(src, trg)
    print(f"{predicts.shape = }")

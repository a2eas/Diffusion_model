import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, voc_size, d_model, nhead, ffn_hidden, nlayers, drop_prob):
        super().__init__()
        self.emb = nn.Embedding(voc_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1000, d_model))  # max_len=1000
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=ffn_hidden,
                                                   dropout=drop_prob)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(self, x):
        x = self.emb(x)  # (batch, seq, d_model)
        x = x + self.pos_enc[:x.size(1), :]  # add positional encoding
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        out = self.encoder(x)
        return out

class Decoder(nn.Module):
    def __init__(self, voc_size, d_model, nhead, ffn_hidden, nlayers, drop_prob):
        super().__init__()
        self.emb = nn.Embedding(voc_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1000, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=ffn_hidden,
                                                   dropout=drop_prob)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=nlayers)
        self.linear = nn.Linear(d_model, voc_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, trg, memory):
        x = self.emb(trg)
        x = x + self.pos_enc[:x.size(1), :]
        x = x.transpose(0, 1)
        out = self.decoder(x, memory)
        out = self.drop(out)
        out = self.linear(out)
        out = out.transpose(0, 1)  # back to (batch, seq, voc_size)
        return out

# Test
batch_size = 2
src_seq_len = 10
trg_seq_len = 8
voc_size = 100
d_model = 64
nlayers = 2
ffn_hidden = 128
nhead = 8
drop_prob = 0.1

src = torch.randint(0, voc_size, (batch_size, src_seq_len))
trg = torch.randint(0, voc_size, (batch_size, trg_seq_len))

encoder = Encoder(voc_size, d_model, nhead, ffn_hidden, nlayers, drop_prob)
decoder = Decoder(voc_size, d_model, nhead, ffn_hidden, nlayers, drop_prob)

enc_out = encoder(src)
out = decoder(trg, enc_out)
print(out.shape)  # (batch_size, trg_seq_len, voc_size)

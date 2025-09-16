import torch
from torch import nn
from .transformer_layers import EncoderLayer,DecoderLayer


class GeneEncoderClass(nn.Module):
    def __init__(self, configs, vocab_size):
        super().__init__()
        self.encoder = Encoder(configs, configs.enc_n_layers)
        self.decoder = Decoder(configs, configs.dec_n_layers)
        self.gene_embeddings = nn.Embedding(vocab_size, configs.hidden_size)
        self.exp_emb = nn.Linear(1, configs.hidden_size)
        self.exp_fc2 = nn.Linear(configs.hidden_size * 2, configs.hidden_size)
        self.LayerNorm = nn.LayerNorm(configs.hidden_size, eps=configs.layer_norm_eps)
        self.drop_out2 = nn.Dropout(p=configs.drop_rate)
        self.src_pad_idx, self.trg_pad_idx = 0, 0
        self.device = configs.device
        self.fc_hint = nn.Linear(configs.hidden_size, configs.feature_dim)
        #self.fc_out_exp = nn.Linear(configs.hidden_size, configs.bin_num)
        self.fc_out_reg = nn.Linear(configs.hidden_size, 1)
    def forward(self, exp, gene_idx , out_gene):
        enc_src, src_mask = self.encode_gene(exp, gene_idx)
        trg  = self.gene_embeddings(out_gene) 
        trg_mask = self.make_trg_mask(out_gene)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        hint_fea = self.hint(enc_src) 
        return self.fc_out_reg(output), hint_fea
    
    def encode_gene(self, exp, gene_idx):
        g_emb = self.gene_embeddings(gene_idx)
        exp_emb = self.exp_emb(exp.unsqueeze(-1))
        src = torch.cat((g_emb, exp_emb), dim = 2)
        src = self.exp_fc2(src)
        src = self.drop_out2(self.LayerNorm(src))
        src_mask = self.make_src_mask(gene_idx)
        enc_src = self.encoder(src, src_mask)
        return enc_src, src_mask
    
    def encode_gene_hint(self, exp, gene_idx):
        enc_src, _ = self.encode_gene(exp, gene_idx)
        hint_fea = self.hint(enc_src) 
        return hint_fea
        #return enc_src[:,0,:]
    
    def hint(self, enc_src):
        hint = self.fc_hint(enc_src[:,0,:])
        return hint
    
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.ones(trg_len, trg_len).type(torch.ByteTensor).to(self.device)
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
class Encoder(nn.Module):

    def __init__(self, configs, gene_n_layers):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hidden_size=configs.hidden_size,
                                                  ffn_hidden=configs.hidden_size,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(gene_n_layers)])

    def forward(self, x, s_mask=None):
        for layer in self.layers:
            x = layer(x, s_mask)

        return x

class Decoder(nn.Module):
    def __init__(self, configs, gene_n_layers):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(hidden_size=configs.hidden_size,
                                                  ffn_hidden=configs.hidden_size,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(gene_n_layers)])

        self.linear = nn.Linear(configs.hidden_size, configs.hidden_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
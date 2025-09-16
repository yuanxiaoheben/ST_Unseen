import torch
import torch.nn as nn
import numpy as np
import math

class Encoder(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(hidden_size=configs.hidden_size,
                                                  ffn_hidden=configs.hidden_size,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(configs.gene_n_layers)])

    def forward(self, x, s_mask=None):
        for layer in self.layers:
            x = layer(x, s_mask)

        return x

class Decoder(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(hidden_size=configs.hidden_size,
                                                  ffn_hidden=configs.hidden_size,
                                                  n_head=configs.num_heads,
                                                  drop_prob=configs.drop_rate)
                                     for _ in range(configs.gene_n_layers)])

        self.linear = nn.Linear(configs.hidden_size, configs.hidden_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output

class PositionwiseFeedForward(nn.Module):

    def __init__(self, hidden_size, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden)
        self.linear2 = nn.Linear(hidden, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension. 

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_concat = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, hidden_size]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, hidden_size = tensor.size()

        d_tensor = hidden_size // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, hidden_size]
        """
        batch_size, head, length, d_tensor = tensor.size()
        hidden_size = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, hidden_size)
        return tensor

class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size=hidden_size, n_head=n_head)
        self.norm1 = LayerNorm(hidden_size=hidden_size)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(hidden_size=hidden_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, hidden_size, max_len, device):
        """
        constructor of sinusoid encoding class

        :param hidden_size: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, hidden_size, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, hidden_size, step=2, device=device).float()
        # 'i' means index of hidden_size (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, hidden_size = 512]

        _, seq_len,_ = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, hidden_size = 512]
        # it will add with tok_emb : [128, 30, 512]

class DecoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size=hidden_size, n_head=n_head)
        self.norm1 = LayerNorm(hidden_size=hidden_size)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(hidden_size=hidden_size, n_head=n_head)
        self.norm2 = LayerNorm(hidden_size=hidden_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(hidden_size=hidden_size)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
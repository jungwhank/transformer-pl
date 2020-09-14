import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_sinusoid_encoding_table(n_position, d_model):
    """get sinusoid position encoding"""
    def cal_angle(position, hidden_i):
        return position / np.power(10000, 2 * (hidden_i // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hidden_j) for hidden_j in range(d_model)]
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i + 1
    return sinusoid_table


def get_attn_pad_mask(seq_q, seq_k, padding_idx):
    """get attention pad mask"""
    bs, len_q = seq_q.size()
    bs, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(padding_idx).unsqueeze(1).expand(bs, len_q, len_k)
    return pad_attn_mask


def get_attn_subsequent_mask(seq):
    """get attention subsequent(decoder) mask"""
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.config = config

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.config['d_k']) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.W_Q = nn.Linear(self.config['d_model'], self.config['d_k']*self.config['n_heads'])
        self.W_K = nn.Linear(self.config['d_model'], self.config['d_k']*self.config['n_heads'])
        self.W_V = nn.Linear(self.config['d_model'], self.config['d_v']*self.config['n_heads'])

        self.attention = ScaledDotProductAttention(self.config)

        self.linear = nn.Linear(self.config['d_v']*self.config['n_heads'], self.config['d_model'])
        self.dropout = nn.Dropout(self.config['dropout'])
        self.layer_norm = nn.LayerNorm(self.config['d_model'], eps=1e-6)

    def forward(self, Q, K, V, attn_mask):
        residual, bs = Q, Q.size(0)
        # |Q| = (bs, len_q, d_model)
        q_s = self.W_Q(Q).view(bs, -1, self.config['n_heads'], self.config['d_k']).transpose(1, 2) # |q_s| = (bs, len, n_heads, d_k) => (bs, n_heads, len, d_k)
        k_s = self.W_K(K).view(bs, -1, self.config['n_heads'], self.config['d_k']).transpose(1, 2)
        v_s = self.W_V(V).view(bs, -1, self.config['n_heads'], self.config['d_v']).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config['n_heads'], 1, 1)

        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.config['n_heads']*self.config['d_v'])
        output = self.dropout(self.linear(context))
        output = self.layer_norm(output + residual)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    """two feed forward layers"""
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Conv1d(config['d_model'], config['d_ff'], kernel_size=1)
        self.w_2 = nn.Conv1d(config['d_ff'], config['d_model'], kernel_size=1)
        self.layer_norm = nn.LayerNorm(config['d_model'], eps=1e-6)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        residual = x
        output = F.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class EncoderLayer(nn.Module):
    """Encoder Layer"""
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)
        self.layer_norm = nn.LayerNorm(config['d_model'], eps=1e-6)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_self_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_self_outputs = self.layer_norm(enc_self_outputs + enc_inputs)

        ffn_outputs = self.pos_ffn(enc_self_outputs) # |enc_outputs| = (bs, len, d_model)
        ffn_outputs = self.layer_norm(ffn_outputs + enc_self_outputs)
        return ffn_outputs, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.src_emb = nn.Embedding(self.config['src_vocab_size'], self.config['d_model'])
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config['max_seq_length']+1, self.config['d_model']))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config['n_layers'])])

    def forward(self, enc_inputs):
        # |enc_inputs| = (bs, len)
        bs, src_len = enc_inputs.size(0), enc_inputs.size(1)
        positions = torch.arange(src_len, device=enc_inputs.device, dtype=enc_inputs.dtype).expand(bs, src_len).contiguous() + 1
        pos_mask = enc_inputs.eq(self.config['padding_idx'])
        positions.masked_fill_(pos_mask, 0)

        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(positions)

        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.config['padding_idx'])

        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    """Decoder Layer"""
    def __init__(self, config):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(config)
        self.layer_norm = nn.LayerNorm(config['d_model'], eps=1e-6)
        self.dec_enc_attn = MultiHeadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_self_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_self_outputs = self.layer_norm(dec_self_outputs + dec_inputs)

        dec_enc_outputs, dec_enc_attn = self.dec_enc_attn(dec_self_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_outputs = self.layer_norm(dec_enc_outputs + dec_self_outputs)

        ffn_outputs = self.pos_ffn(dec_enc_outputs)
        ffn_outputs = self.layer_norm(ffn_outputs + dec_enc_outputs)

        return ffn_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.tgt_emb = nn.Embedding(self.config['tgt_vocab_size'], self.config['d_model'])
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config['max_seq_length']+1, self.config['d_model']))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table,freeze=True)
        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config['n_layers'])])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        bs, tgt_len = dec_inputs.size(0), dec_inputs.size(1)
        positions = torch.arange(tgt_len, device=dec_inputs.device, dtype=dec_inputs.dtype).expand(bs, tgt_len).contiguous() + 1
        pos_mask = enc_inputs.eq(self.config['padding_idx'])
        positions.masked_fill_(pos_mask, 0)

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(positions)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config['padding_idx'])
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config['padding_idx'])

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.tgt_proj = nn.Linear(self.config['d_model'], self.config['tgt_vocab_size'], bias=False)

        if self.config['share_enc_dec_weights']:
            print("encoder and decoder embedding weights sharing")
            self.encoder.src_emb.weight = self.decoder.tgt_emb.weight
        if self.config['share_dec_proj_weights']:
            print("decoder embedding and final linear layer weights sharing")
            self.tgt_proj.weight = self.decoder.tgt_emb.weight

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        logits = self.tgt_proj(dec_outputs)
        return logits.view(-1, logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

    def encode(self, enc_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        return enc_outputs, enc_self_attns

    def decode(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_outputs = self.tgt_proj(dec_outputs)
        return dec_outputs
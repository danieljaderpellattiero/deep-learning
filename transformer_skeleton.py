import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=64):
    super(PositionalEncoding, self).__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
              -math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    seq_len = x.size(1)
    x = x + self.pe[:, :seq_len, :].to(x.device)
    return x

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, nhead, dropout=0.1):
    super(MultiHeadAttention, self).__init__()
    assert d_model % nhead == 0, "d_model must be divisible by nhead"

    self.d_model = d_model
    self.nhead = nhead
    self.d_k = d_model // nhead
    self.dropout = nn.Dropout(dropout)
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)
    self.w_o = nn.Linear(d_model, d_model)

  def scaled_dot_product_attention(self, q, k, v, attn_mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    if attn_mask is not None:
      scores = scores.masked_fill(attn_mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    return torch.matmul(attn_weights, v), attn_weights

  def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
    batch_size = query.size(0)

    q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
    k = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
    v = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

    attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, attn_mask)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    return self.w_o(attn_output)

class TransformerEncoderLayer(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
    super(TransformerEncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

  def forward(self, src, src_mask=None, src_key_padding_mask=None):
    src2 = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask,
                          key_padding_mask = src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    src2 = self.linear2(self.dropout1(self.linear1(src)))
    src =  src + self.dropout2(src2)
    src =  self.norm2(src)
    return src

class TransformerEncoder(nn.Module):
  def __init__(self, encoder_layer, num_layers, norm=None):
    super(TransformerEncoder, self).__init__()
    self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    self.norm = norm

  def forward(self, src, mask=None, src_key_padding_mask=None):
    output = src
    for mod in self.layers:
      output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
    if self.norm is not None:
      output = self.norm(output)
    return output

class TransformerModel(nn.Module):
  def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_classes, dim_feedforward=2048, dropout=0.1,
               max_len=512):
    super(TransformerModel, self).__init__()
    self.embedding = nn.Embedding(vocab_size, d_model)
    self.pos_encoder = PositionalEncoding(d_model, max_len)
    encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
    self.fc = nn.Linear(d_model, num_classes)
    self.d_model = d_model

  def forward(self, input_ids, attention_mask=None):
    embedded = self.embedding(input_ids) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    embedded = self.pos_encoder(embedded)
    transformer_out = self.transformer_encoder(embedded)
    output = transformer_out.mean(dim=1)
    return self.fc(output)

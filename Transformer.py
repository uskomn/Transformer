import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# 缩放点积计算
def scaled_dot_product_attention(Q, K, V, mask=None):
    embeded_size=Q.size(-1)
    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(embeded_size)
    if mask is not None:
        scores = scores.masked_fill(mask==0, float('-inf'))
    attention_weights=F.softmax(scores,dim=-1)
    outputs=torch.matmul(attention_weights,V)
    return outputs,attention_weights

# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        super(MultiHeadAttention,self).__init__()

        assert d_model%h==0

        self.d_model=d_model
        self.h=h
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        self.fc_out=nn.Linear(d_model,d_model)

    def forward(self,q,k,v,mask=None):
        batch_size=q.size(0)

        seq_len_q=q.size(1)
        seq_len_k=k.size(1)

        Q=self.w_q(q).view(batch_size,seq_len_q,self.h,-1).transpose(1,2)
        K=self.w_k(k).view(batch_size,seq_len_k,self.h,-1).transpose(1,2)
        V=self.w_v(v).view(batch_size,seq_len_k,self.h,-1).transpose(1,2)

        scaled_attention,_=scaled_dot_product_attention(Q,K,V,mask)
        concat=scaled_attention.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        out=self.fc_out(concat)
        return out

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 残差链接
class ResidualConnection(nn.Module):
    def __init__(self,dropout=0.1):
        super(ResidualConnection,self).__init__()
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,subLayer):
        return x+self.dropout(subLayer(x))

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self,feature_size,epsilon=1e-9):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(feature_size))
        self.beta=nn.Parameter(torch.zeros(feature_size))
        self.epsilon=epsilon

    def forward(self,x):
        mean=x.mean(dim=-1, keepdim=True)
        std=x.std(dim=-1, keepdim=True)
        return self.gamma*(x-mean)/(std+self.epsilon) + self.beta

# add & norm
class SublayerConnection(nn.Module):
    def __init__(self,feature_size,dropout=0.1,epsilon=1e-9):
        super(SublayerConnection,self).__init__()
        self.norm=LayerNorm(feature_size,epsilon)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,subLayer):
        return self.norm(x+self.dropout(subLayer(x)))

# 嵌入层
class Embeddings(nn.Module):
    def __init__(self,vocab_size,d_model):
        super(Embeddings,self).__init__()
        self.embeded=nn.Embedding(vocab_size,d_model)
        self.scale_factor=math.sqrt(d_model)

    def forward(self,x):
        return self.embeded(x)*self.scale_factor

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)

        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)

        div_term=torch.exp(
            torch.arange(0,d_model,2)*(-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe=pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self,x):
        x=x+self.pe[:,:x.size(1),:]
        return self.dropout(x)

# 编码器输入处理
class SourceEmbedding(nn.Module):
    def __init__(self, src_vocab_size, d_model, dropout=0.1):
        super(SourceEmbedding, self).__init__()
        self.embed = Embeddings(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self,x):
        x=self.embed(x)
        return self.positional_encoding(x)

# 解码器输入处理
class TargetEmbedding(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, dropout=0.1):
        super(TargetEmbedding, self).__init__()
        self.embed = Embeddings(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

    def forward(self,x):
        x=self.embed(x)
        return self.positional_encoding(x)

def create_padding_mask(seq,pad_token=0):
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask

def create_look_ahead_mask(size):
    mask = torch.tril(torch.ones(size, size)).type(torch.bool)  # 下三角矩阵
    return mask

def create_decoder_mask(tgt_seq, pad_token=0):
    padding_mask = create_padding_mask(tgt_seq, pad_token)
    look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1)).to(tgt_seq.device)

    combined_mask = look_ahead_mask.unsqueeze(0) & padding_mask
    return combined_mask

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self,d_model,h,d_ff,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.self_atten=MultiHeadAttention(d_model,h)
        self.feed_forward=FeedForward(d_model,d_ff,dropout)

        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self,x,src_mask):
        x = self.sublayers[0](x, lambda x: self.self_atten(x, x, x, src_mask))
        x = self.sublayers[1](x, self.feed_forward)
        return x

# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, h)
        self.cross_attn = MultiHeadAttention(d_model, h)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.sublayers = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])
        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        x = self.sublayers[2](x, self.feed_forward)

        return x

# 编码器
class Encoder(nn.Module):
    def __init__(self, d_model, N, h, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 解码器
class Decoder(nn.Module):
    def __init__(self, d_model, N, h, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, h, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.src_embedding = SourceEmbedding(src_vocab_size, d_model, dropout)
        self.tgt_embedding = TargetEmbedding(tgt_vocab_size, d_model, dropout)

        self.encoder = Encoder(d_model, N, h, d_ff, dropout)
        self.decoder = Decoder(d_model, N, h, d_ff, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_mask = create_padding_mask(src)
        tgt_mask = create_decoder_mask(tgt)

        enc_output = self.encoder(self.src_embedding(src), src_mask)

        dec_output = self.decoder(self.tgt_embedding(tgt), enc_output, src_mask, tgt_mask)

        output = self.fc_out(dec_output)

        return output
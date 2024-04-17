
import os
import sys
sys.path.append(os.getcwd())
import torch
from torch import nn
from net.module.attention.attention2d import MultiHeadAttention
import math
from torch.autograd import Variable
class RotaryPositionalEmbeddings(nn.Module):
    """
    ---
    title: Rotary Positional Embeddings (RoPE)
    summary: >
      Annotated implementation of RoPE from paper
      RoFormer: Enhanced Transformer with Rotary Position Embedding
    ---

    # Rotary Positional Embeddings (RoPE)

    This is an implementation of
    [Rotary Positional Embeddings (RoPE)](https://arxiv.org/abs/2104.09864)
    in [PyTorch](https://pytorch.org).

    Rotary Positional Embeddings (RoPE) encode position information of tokens
    with a rotation matrix that naturally incorporates explicit relative position
    dependency.

    Here's [the training code](experiment.html) for training a transformer model with RoPE
     on Tiny Shakespeare dataset.
    """
    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        """
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        # Get sequence length
        seq_len = x.shape[0]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]
        neg_half_x = self._neg_half(x_rope)

        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

        #
        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    """
    ## Multi-head attention with rotary positional embeddings

    We override [multi-head attention from original transformer](../mha.html).
    """

    def __init__(self, heads: int, d_model: int, rope_percentage: float = 0.5, dropout_prob: float = 0.0):
        super().__init__(heads, d_model, dropout_prob)

        # Rotary positional embedding layers
        d_rope = int(self.d_k * rope_percentage)
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        """

        # Calculate dot-product with RoPE
        return torch.einsum('ibhd,jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))

class SinusoidalPositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False)
        return self.dropout(x)

class AbsolutePositionEmbedding(nn.Module):
    '''
    absolute position embedding
    '''
    def __init__(self,T,D) -> None:
        '''
        T:seq_len
        D:d_dim
        '''
        super().__init__()
        self.position_emb=nn.Parameter(torch.randn(1,T,D))
    def forward(self,x):
        x=x+self.position_emb
        return x
if __name__=='__main__':
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
    x = x[:, None, None, :]
    # print(x.shape)
    rotary_pe = RotaryPositionalEmbeddings(4)
    # print(rotary_pe(x).shape)
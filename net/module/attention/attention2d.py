import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from einops import rearrange
from torch import nn, einsum
import math
from typing import Optional, List

import torch
from torch import nn
from torch.nn import init

class PrepareForMultiHeadAttention(nn.Module):


    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]` or `[batch_size, d_model]`.
        # We apply the linear transformation to the last dimension and split that into
        # the heads.
        head_shape = x.shape[:-1]

        # Linear transform
        x = self.linear(x)

        # Split last dimension into heads
        x = x.view(*head_shape, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]` or `[batch_size, heads, d_model]`
        return x


class MultiHeadAttention(nn.Module):
    """
    ---
    title: Multi-Headed Attention (MHA)
    summary: >
      This implements the Multi-Headed Attention used in transformers
      using PyTorch with explanations.
    ---

    # Multi-Headed Attention (MHA)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/basic/autoregressive_experiment.ipynb)

    This is a tutorial/implementation of multi-headed attention
    from paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
    in [PyTorch](https://pytorch.org/).
    The implementation is inspired from [Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html).

    Here is the [training code](basic/autoregressive_experiment.html) that uses a basic transformer
    with MHA for NLP auto-regression.

    [Here is an experiment implementation](basic/autoregressive_experiment.html) that trains a simple transformer.
    """
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        """
        * `heads` is the number of heads.
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=1)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', query, key)

    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        """
        `mask` has shape `[seq_len_q, seq_len_k, batch_size]`, where first dimension is the query dimension.
        If the query dimension is equal to $1$ it will be broadcasted.
        """

        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        # Same mask applied to all heads.
        mask = mask.unsqueeze(-1)

        # resulting mask has shape `[seq_len_q, seq_len_k, batch_size, heads]`
        return mask

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        x = rearrange(x,'b n d -> n b d')
        seq_len, batch_size, _ = x.shape

        if mask is not None:
            mask = self.prepare_mask(mask, x.shape, x.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores =scores*self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Save attentions if debugging

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Save attentions for any other calculations 
        # self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        x = self.output(x)
        
        # rehsape
        x = rearrange(x,'n b d -> b n d')
        
        del query,key,value,scores
        
        return x,attn


class   AttentionBlock(nn.Module):
    '''
    Attention gate for image segmentation
    Fl:encoder x channels
    Fg:decoder x channels
    Fint:temporary variable channels
    '''
    def __init__(self,Fl,Fg,Fint) -> None:
        super().__init__()
        self.wg=nn.Sequential(
            nn.Conv2d(in_channels=Fg,out_channels=Fint,stride=1,kernel_size=1),
            nn.BatchNorm2d(Fint)
        )
        self.wx=nn.Sequential(
            nn.Conv2d(in_channels=Fl,out_channels=Fint,stride=1,kernel_size=1),
            nn.BatchNorm2d(Fint)
        )
        self.psi=nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=Fint,out_channels=1,stride=1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    def forward(self,x,g):
        '''
        x:encoder x
        g:decoder x
        '''
        out=self.wx(x)+self.wg(g)
        att=self.psi(out)
        x=att*x
        return x  


# self attetion for 2d image
class MSA(nn.Module):
    def __init__(self,dim,head=8,dropout=0.1) -> None:
        super().__init__()
        assert dim%head==0,"dim必须整除head"
        self.head=head
        self.dim=dim
        self.head_dim=dim//head
        self.scale=self.head**(-0.5)
        self.dropout=nn.Dropout(dropout)
        # qkv
        self.qkv=nn.Linear(dim,3*dim,bias=False)
        self.fc=nn.Linear(dim,dim)

        # self.init_weights()
    def forward(self,x):
        # x:(b,N,d)        
        b,n,_,h=*x.shape,self.head

        # (b,n,3d)->(3,b,n,d)
        qkv=self.qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=h),qkv)

        # attention
        dots=einsum('b h i d, b h j d -> b h i j', q, k)*self.scale
        attn=dots.softmax(dim=-1)
        attn=self.dropout(attn)

        # out
        out=einsum('b h i j, b h j d -> b h i d', attn, v)
        out=rearrange(out,'b h n d -> b n (h d)')
        out=self.fc(out)

        return out,attn
    def init_weights(self):
        '''
        init_weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight,nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MultiHeadAttention(heads=8,d_model=512).to(device)
    data=torch.randn(32,1024,512).to(device)
    out,_=model(data)
    print(out.shape)
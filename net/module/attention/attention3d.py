import torch
import torch.nn as nn
from einops import rearrange
import os
import sys
sys.path.append(os.getcwd())
from utils.utils import printInfo
from net.module.common import *
from net.module.attention.attention2d import *
from net.module.attention.positionEncoding import *
class MultiHeadSelfAttention3D(nn.Module):
    '''
    Normal Self-Attention for Image Series
    '''
    def __init__(self, embed_dim:int, num_heads:int,d_k:int):
        assert embed_dim%num_heads==0,"embed_dim%num_heads must be 0"
        assert d_k%num_heads==0,"d_k%num_heads must be 0"
        super(MultiHeadSelfAttention3D, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim=embed_dim
        self.d_k=d_k
        self.scale=(self.d_k//num_heads) ** -0.5
        
        self.query = nn.Conv3d(in_channels=embed_dim,out_channels=self.d_k,kernel_size=1)
        self.key = nn.Conv3d(in_channels=embed_dim, out_channels=self.d_k,kernel_size=1)
        self.value = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,kernel_size=1)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        '''
        x:(t,b,c,h,w)
        return:(b,thw,c)
        '''

        # (t,b,c,h,w)->(b,c,t,h,w)
        t,b,c,h,w=x.shape
        N=t*h*w
        x=x.permute(1,2,0,3,4)

        # caculate the Q,K,V->(b,thw,c)
        q = self.query(x).reshape(b,c,-1).permute(0,2,1)
        k = self.key(x).reshape(b,c,-1).permute(0,2,1)
        v = self.value(x).reshape(b,c,-1).permute(0,2,1)

        # divide into multi-head (b,thw,c)->(b,heads,thw,c)
        q=q.reshape(b,N,self.num_heads,self.d_k//self.num_heads).permute(0,2,1,3)
        k=k.reshape(b,N,self.num_heads,self.d_k//self.num_heads).permute(0,2,1,3)
        v=v.reshape(b,N,self.num_heads,self.head_dim).permute(0,2,1,3)
    

        attn_weights = torch.matmul(q,k.transpose(-2,-1))*self.scale
        attn_weights = attn_weights.softmax(dim=-1)


        attn_value=torch.matmul(attn_weights,v).transpose(1,2).contiguous().view(b, N,self.embed_dim)
    

        x = self.fc(attn_value)
        
        return x,attn_weights


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)

# 通用Attention头
class ImageAttention(nn.Module):
    def __init__(self,attention_block,reshape_block=None,return_attn=False) -> None:
         super().__init__()
         self.attention_block=attention_block
         self.reshape_block=reshape_block
         self.return_attn=return_attn
         embed_dim=self.attention_block.embed_dim
         self.conv=nn.Conv3d(embed_dim,embed_dim,kernel_size=1)
    def forward(self,x):
        '''
        x:(t,b,c,h,w)
        return:(t,b,c,h,w)
        '''
        h,w=x.shape[-2:]
        x,attn_weights=self.attention_block(x)
        
        # (b,thw,c)->(b,c,t,h,w)
        if self.reshape_block is None:
            x=rearrange(x, "b (t h w) n -> b n t h w", h=h,w=w)
        else:
            x=self.reshape_block(x)
        x=self.conv(x)
        
        # (b,c,t,h,w)->(t,b,c,h,w)
        x=x.permute(2,0,1,3,4)
        if self.return_attn:
            return x,attn_weights
        else:
            return x

# TSAttention
class TSAttention(nn.Module):
    def __init__(self,in_channels, patch_size, emb_size,attention,position_encoding=True,frames=6) -> None:
         super().__init__()
         self.in_channels=in_channels
         self.patch_size = patch_size  
         self.emb_size=emb_size
         self.embedding=PatchEmbedding(in_channels,patch_size,emb_size)
         self.attention=attention
         self.conv=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
         if (not position_encoding) or isinstance(self.attention,RotaryPEMultiHeadAttention):
            self.position_emb=nn.Identity()
         else:
            self.position_emb=AbsolutePositionEmbedding(frames,emb_size)

    def forward(self,x):
        if(not isinstance(x,torch.Tensor)):
            x=torch.stack(x,dim=0)
        T,B,C,H,W=x.shape
        
        # x:(t,b,c,h,w)
        out=rearrange(x,'t b c h w -> (t b) c h w')
        
        # x:(tb,HW,d)
        out=self.embedding(out)
        
        # x->(bHW,t,d)
        out=rearrange(out,'(t b) n d -> (b n) t d',t=T,b=B)
        out=self.position_emb(out)
        # temporal attention
        out,attn=self.attention(out)

        # x->(t,b,c,h,w)
        out=rearrange(out,'(b n) t d -> (t b) n d',t=T,b=B)
        out=rearrange(out,'B (H W) (c h w) -> B c (H h) (W w)',H=H//self.patch_size,h=self.patch_size,w=self.patch_size)
        # out=self.conv(out)
        out=rearrange(out,'(t b) c h w -> t b c h w',t=T,b=B)
        
        # residual
        out=out+x
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
    pass
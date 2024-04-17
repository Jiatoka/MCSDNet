import sys
import os
sys.path.append(os.getcwd())
import torch
from torch import nn
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from timm.models.vision_transformer import Block as ViTBlock
from timm.models.vision_transformer import Attention
from torch.nn import init
class Xception(nn.Module):
    def __init__(self, channel_in) -> None:
        super().__init__()
        self.conv1_1=nn.Conv2d(channel_in,channel_in,kernel_size=1,stride=1)
        self.conv3_3=nn.Sequential(
            nn.Conv2d(channel_in,channel_in//6,kernel_size=1,stride=1),
            nn.Conv2d(channel_in//6,channel_in//6,kernel_size=3,stride=1,padding=1,groups=channel_in//6),
            nn.Conv2d(channel_in//6,channel_in//6,kernel_size=1)
        )
        self.conv5_5=nn.Sequential(
            nn.Conv2d(channel_in,channel_in//2,kernel_size=1,stride=1),
            nn.Conv2d(channel_in//2,channel_in//2,kernel_size=3,stride=1,dilation=2,padding=2,groups=channel_in//2), 
            nn.Conv2d(channel_in//2,channel_in//2,kernel_size=1,stride=1)
        )
        self.concat=nn.Conv2d(channel_in+channel_in//6+channel_in//2,channel_in,stride=1,kernel_size=1)
    def forward(self,x):
        # x:(B,C,H,W)
        x1=self.conv1_1(x)
        x2=self.conv3_3(x)
        x3=self.conv5_5(x)
        x=torch.cat([x1,x2,x3],dim=1)
        x=self.concat(x)
        return x        

class Transformer(nn.Module):
    def __init__(self, dim,seq_len,heads=8, mlp_ratio=4., drop=0.,
     drop_path=0.1,level=2,pos=True) -> None:
        super().__init__()
        self.pos=pos
        self.pos_emb=nn.Parameter(torch.randn(1,seq_len,dim))
        self.layers=nn.ModuleList([ViTBlock(dim=dim,num_heads=heads,mlp_ratio=mlp_ratio, qkv_bias=True,proj_drop=drop,
                         drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm
        ) for _ in range(level)])
        self.level=level
    def forward(self,x):
        # x:(B,n,d)
        if self.pos:
            x=x+self.pos_emb
        for i in range(self.level):
            x=self.layers[i](x)
        return x

class Aggregation(nn.Module):
    def __init__(self,dim,aggregation) -> None:
        super().__init__()
        if aggregation=='concat':
            self.fuse=nn.Conv2d(dim*2,dim,kernel_size=1,stride=1)
        else:
            self.fuse=nn.Identity()
        self.aggregation=aggregation
    def forward(self,a,b):
        # input:(B,C,H,W)
        if self.aggregation=='sum':
            return self.fuse(a+b)
        else:
            return self.fuse(torch.cat([a,b],dim=1))
class STSubBlock(nn.Module):
    '''Spatiotemporal Transformer Block v1'''
    def __init__(self, dim, mlp_ratio=4., drop=0.,
     drop_path=0.1,shape=None,aggregation='sum',temporal=True,spatial=True,mlp_layer=False,num_heads=8,normalize='sequence'):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        T,C,H,W=shape
        self.normalize=normalize
        if self.normalize=='sequence':
            self.norm1=nn.LayerNorm([T*C,H,W])
            self.norm2=nn.LayerNorm([T*C,H,W])
        elif self.normalize=='layer':
            self.norm1=nn.LayerNorm([T*C])
            self.norm2=nn.LayerNorm([T*C])
        else:
            self.norm1=nn.BatchNorm2d(T*C)
            self.norm2=nn.BatchNorm2d(T*C)
        self.spatial_attn=Attention(
            T*C,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
        )
        self.temporal_attn=Attention(
            H*W,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
        )
        self.aggregation=Aggregation(T*C,aggregation)
        reduce=int(mlp_ratio*T*C)
        if mlp_layer:
            self.mlp=Xception(T*C)
        else:
            self.mlp=nn.Sequential(
            nn.Conv2d(T*C,reduce,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(reduce,T*C,kernel_size=1,stride=1),
            nn.Dropout(drop)
        )
        self.temporal=temporal
        self.spatial=spatial
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        '''
        x:输入是 B TC H W
        '''
        B, C, H, W = x.shape
        input_data=None
        # layernorm
        if self.normalize=='layer':
            input_data=rearrange(x,'B C H W ->B (H W) C')
            input_data=self.norm1(input_data)
            input_data=rearrange(input_data,'B (H W) C -> B C H W',H=H)
        else:
            input_data=self.norm1(x)
        if self.spatial: 
            # spatio attn
            spatio_map=rearrange(input_data,'B C H W -> B (H W) C')
            spatio_map=self.spatial_attn(spatio_map)
            spatio_map=rearrange(spatio_map,'B (H W) C -> B C H W',B=B,C=C,H=H,W=W)
        if self.temporal:
            # temporal attn
            temporal_map=rearrange(input_data,'B C H W -> B C (H W)')
            temporal_map=self.temporal_attn(temporal_map)
            temporal_map=rearrange(temporal_map,'B C (H W) -> B C H W',B=B,C=C,H=H,W=W)
        normal_data=None
        if self.temporal and not self.spatial:
            # fuse
            x = x + self.drop_path(temporal_map)
            # layernorm
            if self.normalize=='layer':
                normal_data=rearrange(x,'B C H W ->B (H W) C')
                normal_data=self.norm2(normal_data)
                normal_data=rearrange(normal_data,'B (H W) C -> B C H W',H=H)
            else:
                normal_data=self.norm2(x)
            x = x + self.drop_path(self.mlp(normal_data))
        elif not self.temporal and self.spatial:
            # fuse
            x = x + self.drop_path(spatio_map)
            # layernorm
            if self.normalize=='layer':
                normal_data=rearrange(x,'B C H W ->B (H W) C')
                normal_data=self.norm2(normal_data)
                normal_data=rearrange(normal_data,'B (H W) C -> B C H W',H=H)
            else:
                normal_data=self.norm2(x)
            x = x + self.drop_path(self.mlp(normal_data))
        else:
            # fuse
            x = x + self.drop_path(self.aggregation(spatio_map,temporal_map))
            # layernorm
            if self.normalize=='layer':
                normal_data=rearrange(x,'B C H W ->B (H W) C')
                normal_data=self.norm2(normal_data)
                normal_data=rearrange(normal_data,'B (H W) C -> B C H W',H=H)
            else:
                normal_data=self.norm2(x)
            x = x + self.drop_path(self.mlp(normal_data))
        return x
class ViTSubBlock(ViTBlock):
    """A block of Vision Transformer."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1):
        super().__init__(dim=dim, num_heads=8, mlp_ratio=mlp_ratio, qkv_bias=True,
                         drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight,nonlinearity='relu')
    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):
        '''
        x:输入是 B TC H W
        '''
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    

class AttentionFuse(nn.Module):
    '''
    ScaleDotAttention
    '''
    def __init__(self,d_model) -> None:
        super().__init__()
        # proj
        self.d_model=d_model
        self.query=nn.Linear(d_model,d_model)
        self.key=nn.Linear(d_model,d_model)
        self.value=nn.Linear(d_model,d_model)
        self.scale=1.0/math.sqrt(d_model)

    def forward(self,query,key_value):
        T,B,C,H,W=query.shape
        # reshape
        # T B C H W -> B THW C
        query=rearrange(query,'T B C H W -> B (T H W) C')
        key_value=rearrange(key_value,'T B C H W -> B (T H W) C')

        # scale dot
        q,k,v=self.query(query),self.key(key_value),self.value(key_value)
        
        # bnd,bnd->bnn 

class SumFuse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,query,key_value):
        out=query+key_value
        return out 

class MixPooling2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
    def forward(self,x):
        return self.avg_pool(x)+self.max_pool(x)

class CBAMSpatialAttentionModule(nn.Module):
    '''
    
    '''
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x) :
        # x:(TB C H W)
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output*x

class SpatioAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16):
        super().__init__()
        '''
        dim: (TB) C H W
        '''
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)

        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, dd_k, stride=1, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2*dim, 1)

    def forward(self, x):
        # x: (TB) C H W 
        u = x.clone()
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        
        f_g = self.conv1(attn)
        split_dim = f_g.shape[1] // 2
        f_x, g_x = torch.split(f_g, split_dim, dim=1)
        return torch.sigmoid(g_x) * f_x

class SpatialAttention(nn.Module):
    """A Spatial Attention block for SimVP"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True,dilation=3, reduction=16,spatio_attn='cbam'):
        '''
        d_model: C
        '''
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        if spatio_attn=='cbam':
            self.spatial_gating_unit = CBAMSpatialAttentionModule()
        else: 
            self.spatial_gating_unit = SpatioAttentionModule(d_model, kernel_size,
                                                        dilation=dilation, reduction=reduction)

        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        # x: (TB) C H W
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixMlp(nn.Module):
    def __init__(self,
                 in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)  # 1x1
        self.dwconv = DWConv(hidden_features)                  # CFF: Convlutional feed-forward network
        self.act = act_layer()                                 # GELU
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1) # 1x1
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y

class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8,shortchut=False):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        self.shortcut=shortchut
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)
        if self.shortcut:
            self.residual=nn.Conv2d(C_in, C_out,kernel_size=1,stride=1)
    def forward(self, x):
        if self.shortcut:
            shorcut=self.residual(x)
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        if self.shortcut:
            y=y+shorcut
        return y

class ShuffleChannel(nn.Module):
    '''
    Shuffle the channel order
    '''
    def __init__(self, groups) -> None:
        '''
        groups:the groups of the channel
        output: output[:,i,:,:]=input[:,i%groups*groups_num+i//groups,:,:], i=[0,channels-1],groups_num=channels/groups
                output B (T C) H W ,input B (T C) H W
        '''
        super().__init__()
        self.groups=groups
    def forward(self,x):
        # x:B TC H W
        B,C,H,W=x.shape
        assert C%self.groups==0,"C cannot be evenly divided by self.groups"
        x=rearrange(x,"B (T C) H W -> B (C T) H W",T=self.groups)
        return x
class ReconstructChannel(nn.Module):
    '''
    Reconstruct the channel order
    '''
    def __init__(self,groups) -> None:
        super().__init__()
        self.groups=groups
    def forward(self,x):
        # x: B CT H W
        B,C,H,W=x.shape
        assert C%self.groups==0,"C cannot be evenly divided by self.groups"
        x=rearrange(x,"B (C T) H W -> B (T C) H W",T=self.groups)
        return x
class MixConv2d(nn.Module):
    '''
    Mix Convolutional is a group convolutinonal which shuffle the batch along with the T*C(channel) dimesion.After shuffling 
    the images, the same channel at different times in the image sequence can be adjacent.Then,the convolutinaol kernel can
    effectively extract time information by group.The groups is the number of frames.
    '''
    def __init__(self, in_channels: int,
                    out_channels: int,
                    kernel_size: int,
                    stride: int = 1,
                    padding: int = 0,
                    dilation: int = 1 ,
                    timesteps=6,
                    bias: bool = True,
                    padding_mode: str = 'zeros',
                    groups=6) -> None:
        '''
        timesteps:the number of 
        '''
        super().__init__()
        self.shuffle=ShuffleChannel(groups=timesteps)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.reconstruct=ReconstructChannel(groups=timesteps)
    def forward(self,x):
        # x:(B,TC,H,W)
        x=self.shuffle(x)
        x=self.conv(x)
        x=self.reconstruct(x)
        return x

class GASubBlock(nn.Module):
    """A GABlock (gSTA) for SimVP,just like the normal transformer block"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = TemporalAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class TemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""

    def __init__(self, d_model, kernel_size=21, attn_shortcut=True,dilation=3, 
                reduction=16,dynamic='avg',timesteps=6,conv='standard',groups=6):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.activation = nn.GELU()                          # GELU
        self.spatial_gating_unit = TemporalAttentionModule(d_model, kernel_size,dilation=dilation,reduction=reduction,
        dynamic=dynamic,timesteps=timesteps,conv=conv,groups=groups)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)         # 1x1 conv
        self.attn_shortcut = attn_shortcut

    def forward(self, x):
        if self.attn_shortcut:
            shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        if self.attn_shortcut:
            x = x + shortcut
        return x
    
class TemporalAttentionModule(nn.Module):
    """Large Kernel Attention for SimVP"""

    def __init__(self, dim, kernel_size, dilation=3, reduction=16,
                dynamic='avg',timesteps=6,conv='standard',groups=6):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = (dilation * (dd_k - 1) // 2)
        if conv=='standard':
            self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p)
            self.conv_spatial = nn.Conv2d(
                dim, dim, dd_k, stride=1, padding=dd_p, dilation=dilation)
        elif conv=='mix':
            self.conv0 = MixConv2d(dim, dim, d_k, padding=d_p, timesteps=timesteps,groups=groups)
            self.conv_spatial = MixConv2d(dim, dim, dd_k,
                             stride=1, padding=dd_p, timesteps=timesteps, dilation=dilation,groups=groups)
        else:
            raise Exception(f'error!!!No convolutional name is {conv}')
        self.conv1 = nn.Conv2d(dim, dim, 1)

        self.reduction = max(dim // reduction, 4)
        if dynamic=='avg':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif dynamic=='max':
            self.avg_pool = nn.AdaptiveMaxPool2d(1)
        elif dynamic=='mix':
            self.avg_pool = MixPooling2D()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // self.reduction, bias=False), # reduction
            nn.ReLU(True),
            nn.Linear(dim // self.reduction, dim, bias=False), # expansion
            nn.Sigmoid()
        )

    def forward(self, x):
        u = x.clone()
        # statical attention 
        attn = self.conv0(x)           # depth-wise conv
        attn = self.conv_spatial(attn) # depth-wise dilation convolution
        f_x = self.conv1(attn)         # 1x1 conv
        
        # append a se operation
        # dynamic attention
        b, c, _, _ = x.size()
        se_atten = self.avg_pool(x).view(b, c)
        se_atten = self.fc(se_atten).view(b, c, 1, 1)

        # fuse
        return se_atten * f_x * u

class TAUSubBlock(GASubBlock):
    """A TAUBlock (tau) for Temporal Attention Unit"""

    def __init__(self, dim, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU,attn_shortcut=True):
        super().__init__(dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                 drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        
        self.attn = TemporalAttention(dim, kernel_size,attn_shortcut=attn_shortcut)

class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0,shortcut=False,attn_shortcut=True):
        super(MetaBlock, self).__init__()
        '''
        in_channels: T*C,the attention input channels
        out_channels: T*C,the attention output channels
        '''
        self.shortcut=shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU,attn_shortcut=attn_shortcut)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if self.shortcut:
            self.residual=nn.Conv2d(in_channels, out_channels,kernel_size=1,stride=1)
    def forward(self, x):
        z = self.block(x)
        if self.in_channels != self.out_channels:
            z=self.reduction(z)
        if self.shortcut:
            z=z+self.residual(x)
        return  z

class SpatioTemporalAttention(nn.Module):
    """A Temporal Attention block for Temporal Attention Unit"""
    def __init__(self, in_channels,d_model,timesteps, kernel_size=21, attn_shortcut=True,dilation=3,
                reduction=16,dynamic='avg',fuse='sum',conv='standard',spatio_attn='default',temporal_attn='default',groups=6):
        '''
        in_channels: input channel
        d_model: T * C
        timestep: the number of frames
        '''
        assert in_channels*timesteps==d_model ,"in_channels*timesteps must be equal to d_model"
        super().__init__()
        if spatio_attn=='None':
            self.spatio_encoder=nn.Identity()
        else:
            self.spatio_encoder=SpatialAttention(
            d_model=in_channels,
            kernel_size=kernel_size,
            attn_shortcut=attn_shortcut,
            dilation=dilation,
            reduction=reduction,
            spatio_attn=spatio_attn
        )

        if temporal_attn=='None':
            self.temporal_encoder=nn.Identity()
        else:
            self.temporal_encoder=TemporalAttention(d_model=d_model,kernel_size=kernel_size,
                                                attn_shortcut=attn_shortcut,
                                                dilation=dilation,reduction=reduction,
                                                dynamic=dynamic,timesteps=timesteps,
                                                conv=conv,groups=groups)
        if fuse=='attn':
            self.fuse=AttentionFuse(in_channels)
        elif fuse=='sum':
            self.fuse=SumFuse()
        else:
            raise Exception(f'SpatioTemporal Attention meet error,there is not fuse method {fuse}')
        self.attn_shortcut=attn_shortcut
    def forward(self,x):
        # X:T B C H W
        T,B,C,H,W = x.shape
        # T,B,C,H,W -> TB,C,H,W
        spatio_feature_map=rearrange(x,'T B C H W -> (T B) C H W')
        spatio_feature_map=self.spatio_encoder(spatio_feature_map)
        spatio_feature_map=rearrange(spatio_feature_map,'(T B) C H W -> T B C H W',T=T,B=B)
        # reshape x to (B ,TC ,H ,W)
        temporal_feature_map=rearrange(x,'T B C H W -> B (T C) H W')
        temporal_feature_map=self.temporal_encoder(temporal_feature_map)
        temporal_feature_map=rearrange(temporal_feature_map,'B (T C) H W -> T B C H W',T=T,C=C)
        out=self.fuse(spatio_feature_map,temporal_feature_map)
        if self.attn_shortcut:
            out=out+x
        return out        

class STAUSubBlock(GASubBlock):
    '''
    A STAUBlock (tau) for Spatio-Temporal Attention Unit
    It is a  multi-branch network
    '''
    def __init__(self,in_channels,dim,timesteps, kernel_size=21, mlp_ratio=4.,
                 drop=0., drop_path=0.1, init_value=1e-2, act_layer=nn.GELU,
                 dilation=3,reduction=16,dynamic='avg',fuse='sum',conv='standard',
                 attn_shorcut=True,spatio_attn='default',temporal_attn='default',groups=6):
        '''
        in_channels: image channels
        dim: T*C
        timesteps: the number of frames
        mlp_ratio: the fc ratio
        kernel_size: the kernel size in TAU and  SAU
        groups: the group number of the MixConv2D
        '''
        super().__init__(dim=dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio,
                 drop=drop, drop_path=drop_path, init_value=init_value, act_layer=act_layer)
        self.attn = SpatioTemporalAttention(in_channels,dim, timesteps,kernel_size,
                                            attn_shortcut=attn_shorcut,dilation=dilation,
                                            reduction=reduction,dynamic=dynamic,
                                            fuse=fuse,conv=conv,spatio_attn=spatio_attn,
                                            temporal_attn=temporal_attn,
                                            groups=groups
                                )
    def forward(self,x):
        # x:(T B C H W):
        T,B,C,H,W=x.shape

        # (T B C H W) -> (B TC H W)
        x = rearrange(x,'T B C H W -> B (T C) H W')
        norm1 = self.norm1(x)
        norm1 = rearrange(norm1,'B (T C) H W -> T B C H W',T=T,C=C)
        attn  = self.attn(norm1)
        attn  = rearrange(attn,'T B C H W -> B (T C) H W')
        # B (T C) H W                           
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * attn)
    
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        
        x = rearrange(x,'B (T C) H W -> T B C H W',T=T,C=C)
        return x


if __name__=='__main__':
    pass
import os
import sys
sys.path.append(os.getcwd())
import torch
from torch import nn
from net.module.common import DownSample,Conv2DModule
from net.module.inception import *
from einops import rearrange
from net.module.convlstm.convlstm import *
class BottomConvLSTM(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.in_conv=nn.Identity()
        self.convlstm=ConvLSTMCell(input_size=channel,hidden_size=128)
        self.out_conv=nn.Conv2d(128,channel,kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        # x:(T B C H W)
        T,B,C,H,W=x.shape
        x=rearrange(x,'T B C H W -> (T B) C H W')
        x=self.in_conv(x)
        x=rearrange(x,'(T B) C H W -> T B C H W',T=T)
        h,c=self.convlstm.init_hidden(B,(H,W))
        x=torch.unbind(x)
        out=[]
        for i in range(len(x)):
            h,c=self.convlstm(x[i],h,c)
            out.append(h)
        out=torch.stack(out)
        out=rearrange(out,'T B C H W -> (T B) C H W')
        out=self.out_conv(out)
        out=rearrange(out,'(T B) C H W -> T B C H W',T=T,B=B)
        return out
class BottomConv(nn.Module):
    def __init__(self, in_channels,out_channels,num_groups=4) -> None:
        super().__init__()
        self.down=DownSample()
        self.conv=Conv2DModule(in_channels=in_channels,
                            out_channels=out_channels,
                            batch=False,
                            group_norm=True,
                            num_groups=num_groups)
    def forward(self,x):
        return self.conv(self.down(x))
class BottomConv3D(nn.Module):
    def __init__(self,in_channel,out_channel) -> None:
        super().__init__()
        self.in_conv=nn.Conv2d(in_channel,256,kernel_size=3,stride=1,padding=1)
        self.out_conv=nn.Conv2d(256,out_channel,kernel_size=3,stride=1,padding=1)
        self.conv=nn.Sequential(
            nn.Conv3d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256,256,kernel_size=3,stride=1,padding=1)
            )
    def forward(self,x):
        # x:(T B C H W)
        T,B,C,H,W=x.shape
        x=rearrange(x,'T B C H W -> (T B) C H W')
        x=self.in_conv(x)
        x=rearrange(x,'(T B) C H W -> B C T H W',B=B,T=T)
        out=self.conv(x)
        x=rearrange(x,'B C T H W -> (T B) C H W')
        x=self.out_conv(x)
        x=rearrange(x,'(T B) C H W -> T B C H W',T=T)
        return x

class BottomViT(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 mlp_ratio=4., drop=0.0, drop_path=0.1,
                 shortcut=False):
        '''
            channel_in: (T C)
        '''
        super(BottomViT, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        model_type='vit'
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid,  model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0,shortcut=shortcut)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid,  model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i,shortcut=shortcut))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in,  model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1,shortcut=shortcut))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = rearrange(x,'T B C H W -> B (T C) H W')

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = rearrange(z,'B (T C) H W -> T B C H W',T=T,C=C)
        return y

class STTransformer(nn.Module):
    def __init__(self, channel_in, channel_hid, N2,kernel_size=21,
                 mlp_ratio=4., drop=0.0, drop_path=0.1,
                 shortcut=False,attn_shortcut=True,
                 dynamic='avg',fuse='sum',dilation=3,
                 reduction=16,timesteps=6,conv='standard',spatio_attn='default',
                 temporal_attn='default',groups=6,aggregation=None,shape=[6,128,20,32],
                 temporal=True,spatial=True,mlp_layer=False,pos_emb=False,num_heads=8,normalize='sequence') -> None:
        super().__init__()
        '''
        channel_in:the channels of the image
        channel_hid:the attention channel
        timesteps:the number of the frames
        groups: groups number of the MixConv2D
        '''
        self.pos=pos_emb
        if self.pos:
            self.pos_emb=nn.Parameter(torch.randn(1,shape[0],shape[1],shape[2],shape[3]))
        if aggregation==None:
            aggregation='sum'
        assert N2 >= 2 and mlp_ratio > 1
        if channel_hid==channel_in:
            self.conv1=nn.Identity()
        else:
            self.conv1=nn.Conv2d(channel_in,channel_hid,kernel_size=1,stride=1)
        self.N2 = N2
        self.timesteps=timesteps
        self.channels=channel_in
        # downsample
        enc_layers = [STSubBlock(timesteps*channel_hid,mlp_ratio=mlp_ratio,
        drop=drop,drop_path=drop_path,shape=shape,aggregation=aggregation,temporal=temporal,spatial=spatial,mlp_layer=mlp_layer,num_heads=num_heads,normalize=normalize) for _ in range(N2)]
    
        self.enc = nn.Sequential(*enc_layers)
        if channel_hid==channel_in:
            self.conv2=nn.Identity()
        else:
            self.conv2=nn.Conv2d(channel_hid,channel_in,kernel_size=1,stride=1)
        self.aggregation=aggregation
        self.shortcut=nn.Identity()
        self.concat=nn.Conv2d(2*channel_in,channel_in,stride=1,kernel_size=1,padding=0)
    def forward(self,x):
        T, B, C, H, W = x.shape
        shortcut=self.shortcut(x)
        x = rearrange(x,'T B C H W -> (T B) C H W')
        x = self.conv1(x)
        # pos embedding
        if self.pos:
            x = rearrange(x,'(T B) C H W -> B T C H W',T=T)
            x = x + self.pos_emb
            x = rearrange(x,'B T C H W -> (T B) C H W') 
        x = rearrange(x,'(T B) C H W -> B (T C) H W',T=T)
        for i in range(self.N2):
            x = self.enc[i](x)
        x = rearrange(x,'B (T C) H W -> (T B) C H W',T=T)
        x = self.conv2(x)
        x = rearrange(x,'(T B) C H W -> T B C H W',T=T) 
        return x
if __name__=='__main__':
    pass
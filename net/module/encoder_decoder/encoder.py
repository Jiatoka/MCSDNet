import os
import sys
sys.path.append(os.getcwd())
import torch
from torch import nn
from net.module.common import Conv2DModule,CropAndConcat,UpSample
from einops import rearrange
from utils.utils import printInfo
from torch.nn import init
from typing import List,Union
from net.module.common import ASPP,PPM,MultiScalePool
class DownSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sample=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        return self.sample(x)

class UnetEncoder(nn.Module):
    '''
    navie Unet
    '''
    def __init__(self,in_channels:int,layer_channels:List[int],batch=False,interpolation=False,residual=False,group_norm=False,num_groups=4,aspp=False) -> None:
        super().__init__()
        if not batch:
            group_norm=True
        self.encoder=nn.ModuleList(
            [Conv2DModule(in_channels if i==0 else layer_channels[i-1],layer_channels[i],batch,residual,group_norm,num_groups) for i in range(len(layer_channels))]
        )
        self.down_sample=nn.ModuleList([DownSample() for _ in range(len(layer_channels))])
        self.level=len(layer_channels)
        self.aspp_flag=aspp
        if aspp:
            self.aspp_block=ASPP(layer_channels[-1],layer_channels[-1])
    def forward(self,x:torch.Tensor):
        # x:(tb,c,h,w)
        # encoder
        encode_out=[]
        for i in range(len(self.encoder)):
            x=self.encoder[i](x)
            encode_out.append(x)
            x=self.down_sample[i](x)
        if self.aspp_flag:
            encode_out[-1]=encode_out[-1]+self.aspp_block(encode_out[-1])
        return encode_out
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

class MultiScaleEncoder(nn.Module):
    '''
    MultiScale Unet Encoder
    '''
    def __init__(self,in_channels:int,layer_channels:List[int],batch=False,interpolation=False,residual=False,group_norm=False,num_groups=4,aspp=False) -> None:
        super().__init__()
        if not batch:
            group_norm=True
        aspp=True
        self.encoder=nn.ModuleList(
            [Conv2DModule(in_channels if i==0 else layer_channels[i-1],layer_channels[i],batch,residual,group_norm,num_groups) for i in range(len(layer_channels))]
        )
        self.down_sample=nn.ModuleList([DownSample() for _ in range(len(layer_channels))])
        self.level=len(layer_channels)
        self.aspp_flag=aspp
        if aspp:
            self.aspp_block=ASPP(layer_channels[-1],layer_channels[-1])
        self.ppm=nn.ModuleList([MultiScalePool((20,32),layer_channels[i],layer_channels[-1]) for i in range(len(layer_channels)-1)])
        self.ppm_conv=nn.Conv2d(len(layer_channels)*layer_channels[-1],layer_channels[-1],kernel_size=1)
    def forward(self,x:torch.Tensor):
        # x:(tb,c,h,w)
        # encoder
        encode_out=[]
        for i in range(len(self.encoder)):
            x=self.encoder[i](x)
            encode_out.append(x)
            x=self.down_sample[i](x)
        last_encoder=[]
        for i in range(len(self.encoder)-1):
            last_encoder.append(self.ppm[i](encode_out[i]))
        last_encoder.append(encode_out[-1])
        encode_out[-1]=torch.cat(last_encoder,dim=1)
        encode_out[-1]=self.ppm_conv(encode_out[-1])
        encode_out[-1]=encode_out[-1]+self.aspp_block(encode_out[-1])
        return encode_out
    def get_feature_maps(self,x:torch.Tensor):
        # x:(tb,c,h,w)
        # encoder
        encode_out=[]
        for i in range(len(self.encoder)):
            x=self.encoder[i](x)
            encode_out.append(x)
            x=self.down_sample[i](x)
        last_encoder=[]
        for i in range(len(self.encoder)-1):
            last_encoder.append(self.ppm[i](encode_out[i]))
        last_encoder.append(encode_out[-1])
        encode_out[-1]=torch.cat(last_encoder,dim=1)
        encode_out[-1]=self.ppm_conv(encode_out[-1])

        # multi scale feature maps
        multi_scale_feature_maps=encode_out[-1]
        # encoder feature maps
        encoder_feature_maps=encode_out[:len(self.encoder)-1]
        # align maps
        align_maps=last_encoder[:]
        return multi_scale_feature_maps,encoder_feature_maps,align_maps
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
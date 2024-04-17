import os
import sys
sys.path.append(os.getcwd())
import torch
from torch import nn
from net.module.common import Conv2DModule,CropAndConcat,UpSample
from einops import rearrange
from torch.nn import init
from net.module.encoder_decoder.encoder import *
from net.module.encoder_decoder.bottom import *
from net.module.encoder_decoder.decoder import *
from typing import Union

class MCSDNet(nn.Module):
    def __init__(self,encoder,decoder,bottom:nn.Module):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.bottom=bottom
        self.init_weights()
    def forward(self,x):
        # input x:(t,b,c,h,w)
        T,B,C,H,W=x.shape
        
        # reshape
        x=rearrange(x,'t b c h w -> (t b) c h w')
        encoder_layer_x=self.encoder(x)

        # bottom_x:(tb,c,h,w)->(t,b,c,h,w)
        bottom_x=rearrange(encoder_layer_x[-1],'(t b) c h w -> t b c h w',t=T)
        bottom_x=self.bottom(bottom_x)
        # reshape
        bottom_x=rearrange(bottom_x,'t b c h w -> (t b) c h w')
        encoder_layer_x[-1]=bottom_x
        # decoder
        out1=self.decoder(encoder_layer_x)
        out1=rearrange(out1,'(t b) c h w -> t b c h w',t=T)
        return out1
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



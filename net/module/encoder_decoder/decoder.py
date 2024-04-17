from torch import nn
import torch
from typing import List,Union
from net.module.common import *
from net.module.common import ASPP
class ConvDecoder(nn.Module):
    def __init__(self,in_layer_channels:List[int],out_layer_channels:List[int],out_channel=1,batch=False,residual=False,group_norm=False,num_groups=4,interpolation=False,last_layer=False,level=False) -> None:
        '''
        A Unet Decoder
        last layer: if exist bottom layer
        '''
        super().__init__()
        self.last_layer=last_layer
        self.length=len(in_layer_channels)
        self.up_sample=nn.ModuleList([UpSample(i*2,i,interpolation) for i in 
                                      out_layer_channels])
        self.skip_connection=nn.ModuleList([CropAndConcat() for _ in range(self.length)])
        if self.last_layer==False:
            self.decoder=nn.ModuleList([Conv2DModule(i+o,o,batch,residual,group_norm,num_groups) for i,o in 
                                    zip(in_layer_channels,out_layer_channels)])
            self.decoder[-1]=Conv2DModule(in_layer_channels[-1],out_layer_channels[-1],batch,residual,group_norm,num_groups)
        else:
            self.decoder=nn.ModuleList([Conv2DModule(i+o,o,batch,residual,group_norm,num_groups) for i,o in 
                                    zip(in_layer_channels,out_layer_channels)])
        self.classifier=nn.Conv2d(in_channels=out_layer_channels[0],out_channels=out_channel,kernel_size=1,stride=1)
        self.level=level

        self.init_weights()
    def forward(self,encoder_x:List[torch.tensor],bottom_in=None):
        '''
        encoder:[layer,b,c,h,w]
        bottom_in:(b,c,h,w)
        '''
        if bottom_in is None and self.last_layer==True:
            raise "Decoder doesn't has the bottom input1"
        elif (not bottom_in is None) and self.last_layer==False:
            raise "Decoder doesn't has the bottom input2"
        out_level=[]
        if self.last_layer:
            layer_out=bottom_in
            for i in range(self.length-1,-1,-1):
                layer_out=self.up_sample[i](layer_out)
                layer_out=self.skip_connection[i](encoder_x[i],layer_out)
                layer_out=self.decoder[i](layer_out)
                out_level.insert(0,layer_out)
        else:
            layer_out=encoder_x[-1]
            for i in range(self.length-1,-1,-1):
                if i<self.length-1:
                    layer_out=self.up_sample[i](layer_out)
                    layer_out=self.skip_connection[i](encoder_x[i],layer_out)
                layer_out=self.decoder[i](layer_out)
                out_level.insert(0,layer_out)
        out=self.classifier(out_level[0])
        if self.level:
            return out,out_level
        else:
            return out
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
class UperNet(nn.Module):
    def __init__(self,layer_channels,out_channels,num_classes=1) -> None:
        super().__init__()
        self.fpn=FPNHEAD(layer_channels,out_channels)
        self.segmentaion=nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, kernel_size=3,stride=1,padding=1)
        )
    def forward(self,encoder_x:List[torch.Tensor]):
        # encoder_x:(layer,b,c,h,w)
        x=self.fpn(encoder_x)
        x=self.segmentaion(x)
        return x  
if __name__=='__main__':
    pass
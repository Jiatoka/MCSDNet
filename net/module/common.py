# Different Unet Model
import torch
import os
import sys
sys.path.append(os.getcwd())
import torch.nn.functional as F
import torch.nn as nn
from net.module.attention.attention2d import AttentionBlock
from net.module.convlstm.convlstm import ConvLSTMCell,ResidualConvLSTM
from typing import Union,List
from torch.autograd import Variable
import math
from torch.nn import init
from torchvision.transforms.functional import center_crop 
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()
        self.aspp_conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.aspp_conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.aspp_conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.aspp_conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.aspp_conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[3], dilation=rates[3])

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.aspp_conv1x1_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1x1_output = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)

    def forward(self, x):
        aspp1 = self.aspp_conv1x1_1(x)
        aspp2 = self.aspp_conv3x3_1(x)
        aspp3 = self.aspp_conv3x3_2(x)
        aspp4 = self.aspp_conv3x3_3(x)
        aspp5 = self.aspp_conv3x3_4(x)

        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = self.aspp_conv1x1_5(global_avg_pool)
        global_avg_pool = F.interpolate(global_avg_pool, size=aspp1.shape[2:], mode='bilinear', align_corners=False)

        concatenated = torch.cat([aspp1, aspp2, aspp3, aspp4, global_avg_pool], dim=1)
        output = self.conv1x1_output(concatenated)
        return output
class ResidualBlock(nn.Module):
    '''
    Residual Block
    '''
    def __init__(self,network,in_channels,out_channels,batch=False,group_norm=False,num_groups=4,solid=False) -> None:
        super().__init__()
        self.net=network
        if solid:
            self.conv=nn.Identity()
        elif batch and group_norm==False:
            self.conv=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=1,padding=0),
                nn.BatchNorm2d(out_channels)
            )

        elif batch==False and group_norm:
            self.conv=nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=1,padding=0),
                    nn.GroupNorm(num_groups=num_groups,num_channels=out_channels)
                )
        else:
            self.conv=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=1,padding=0)
            )
        self.activate=nn.ReLU(inplace=True)

        # self.init_weights()
    def forward(self,x):
        out=self.net(x)
        x=self.conv(x)
        x=self.activate(out+x)
        del out
        return x
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

    
class Conv2DModule(nn.Module):
    def __init__(self,in_channels,out_channels,batch=False,residual=False,group_norm=False,num_groups=4) -> None:
        super().__init__()
        if batch:
            self.feature=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1),
                nn.BatchNorm2d(num_features=out_channels),
            )
        elif group_norm:
            self.feature=nn.Sequential(
                    nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1),
                    nn.GroupNorm(num_groups=num_groups,num_channels=out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1),
                    nn.GroupNorm(num_groups=num_groups,num_channels=out_channels),
                ) 
        else:
            self.feature=nn.Sequential(
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=1,kernel_size=3,padding=1),
            )
        if residual:
            self.feature=ResidualBlock(self.feature,in_channels,out_channels,batch,group_norm,num_groups)
            self.activate=nn.Identity()
        else:
            self.activate=nn.ReLU(inplace=True)
        self.residual=residual

        # self.init_weights()
    def forward(self,x):
        out=self.feature(x)
        out=self.activate(out)
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

class DownSample(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sample=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
        return self.sample(x)

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels,interpolation=False) -> None:
        super().__init__()
        self.interpolation=interpolation
        if interpolation:
            self.up=nn.Identity()
            self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1)
        else:
            self.up=nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2)
    def forward(self,x):
        out=self.up(x)
        if self.interpolation:
            out=nn.functional.interpolate(out,scale_factor=2,mode='bilinear')
            out=self.conv(out)
        return out

class CropAndConcat(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,encode_x,decode_x):
        encode_x=center_crop(encode_x,[decode_x.shape[2],decode_x.shape[3]])
        return torch.cat([decode_x,encode_x],dim=1)

class PatchEmbedding(nn.Module):  
    def __init__(self, in_channels, patch_size, emb_size):  
        super().__init__()  
        self.patch_size = patch_size  
        self.projection = nn.Sequential(  
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),  
            nn.Flatten(2)  
        )  
          
        # self.init_weights()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x:torch.Tensor
        x = self.projection(x)
        x=x.permute(0,2,1).contiguous()  
        return x
    def init_weights(self):
        '''
        init_weights
        '''
        for m in self.projection.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight,nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )     
            
    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
            out_puts.append(ppm_out)
        return out_puts
  
class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes = [1, 2, 3, 6],num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes)*self.out_channels, self.out_channels, kernel_size=3,padding=1,stride=1,bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out

class FPN_IN(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class FPN_OUT(nn.Module):
    def __init__(self, in_channels,out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x)

class FPNHEAD(nn.Module):
    def __init__(self, channels=[64,128,256,512], out_channels=256):
        super(FPNHEAD, self).__init__()
        self.level=len(channels)
        self.PPMHead = PPMHEAD(in_channels=channels[-1], out_channels=out_channels)
        self.fpn_layer=nn.ModuleList([])
        for i in range(len(channels)-2,-1,-1):
            self.fpn_layer.insert(
                0,
                nn.ModuleList(
                    [
                        FPN_IN(channels[i],out_channels),
                        FPN_OUT(out_channels,out_channels)
                    ]
                )
            )


        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels*self.level, out_channels, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv_x1 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1)
        
    def forward(self, input_fpn):
        H,W=input_fpn[0].shape[-2:]
        level_out=[]
        # b, 512, 7, 7
        fpn_out = self.PPMHead(input_fpn[-1])
        fpn_out = self.conv_x1(fpn_out)
        level_out.append(F.interpolate(fpn_out, (H,W),mode='bilinear', align_corners=True))
        for i in range(self.level-2,-1,-1):
            fpn_out = nn.functional.interpolate(fpn_out, size=(fpn_out.size(2)*2, fpn_out.size(3)*2),mode='bilinear', align_corners=True)
            fpn_out =  fpn_out + self.fpn_layer[i][0](input_fpn[i])
            fpn_out = self.fpn_layer[i][1](fpn_out)
            if i==0:
                level_out.append(fpn_out)
            else:
                level_out.append(F.interpolate(fpn_out, (H,W),mode='bilinear', align_corners=True))
 
        x = self.fuse_all(torch.cat(level_out, 1))
        
        return x
class MultiScalePool(nn.Module):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(MultiScalePool, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool=nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_sizes),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1),
                )
                 
    def forward(self, x):
        out=self.pool(x)
        return out
        
if __name__=='__main__':
    pass

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())
from utils.utils import printInfo
import torch.nn.functional as F
from typing import Union
class ConvLSTMCell(nn.Module):
    '''
    input_size:input image channels
    hidden_size:the cell state and hidden state channels
    kernel_size:the size of Convolutional Kernel
    batch_norm:if batchnorm2d,default=True
    '''
    def __init__(self,input_size:int,hidden_size:int,kernel_size=3,batch_norm=True,group_norm=False,num_groups=4) -> None:
        super().__init__()
        if not batch_norm:
            group_norm=False
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.kernel_size=kernel_size
        self.pad=kernel_size//2

        self.hidden_lin=nn.Conv2d(in_channels=hidden_size,out_channels=4*hidden_size,kernel_size=kernel_size,stride=1,padding=self.pad)
        self.input_lin=nn.Conv2d(in_channels=input_size,out_channels=4*hidden_size,kernel_size=kernel_size,stride=1,padding=self.pad,bias=False)
        
        if batch_norm:
            if not group_norm:
                self.batch_norm=nn.ModuleList([nn.BatchNorm2d(num_features=hidden_size) for _ in range(4)])
                self.batch_norm_c=nn.BatchNorm2d(num_features=hidden_size)
            else:
                self.batch_norm=nn.ModuleList([nn.GroupNorm(num_groups=num_groups,num_channels=hidden_size) for _ in range(4)])
                self.batch_norm_c=nn.GroupNorm(num_groups=num_groups,num_channels=hidden_size)
        else:
            self.batch_norm=nn.ModuleList([nn.Identity() for _ in range(4)])
            self.batch_norm_c=nn.Identity()
    def init_hidden(self,batch_size,image_size):
        '''
        return h and c
        '''
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_size, height, width, device=self.input_lin.weight.device),
                torch.zeros(batch_size, self.hidden_size, height, width, device=self.input_lin.weight.device))
    def forward(self,x:torch.Tensor,h:Union[torch.Tensor,None]=None,c:Union[torch.Tensor,None]=None):
        if h is None:
            h,c=self.init_hidden(x.shape[0],x.shape[-2:])
        figo=self.hidden_lin(h)+self.input_lin(x)
        figo = torch.split(figo, self.hidden_size, dim=1)
        figo=[self.batch_norm[i](figo[i]) for i in range(4)]
        f,i,g,o=figo        
        c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        h=torch.sigmoid(o)*torch.tanh(self.batch_norm_c(c))
        del figo
        del i,f,g,o
        return h,c

class ResidualConvLSTM(nn.Module):
    '''
    add residual to convlstm
    '''
    def __init__(self,input_size:int,hidden_size:int,kernel_size=3,batch_norm=True,group_norm=False,num_groups=4) -> None:
        super().__init__()
        if not batch_norm:
            group_norm=False
        self.convlstm=ConvLSTMCell(input_size,hidden_size,kernel_size,batch_norm,group_norm,num_groups)
        self.residual=nn.Identity()
    def forward(self,x:torch.Tensor,h:torch.Tensor,c:torch.Tensor):
        '''
        x:(b,c,h,w)
        return:residual output,h,c
        '''
        h,c=self.convlstm(x,h,c)
        output=self.residual(x)+h
        return output,h,c
    def init_hidden(self,batch_size,image_size):
        '''
        return h and c
        '''
        return self.convlstm.init_hidden(batch_size,image_size)
from typing import List

class ConvLSTM(nn.Module):
    def __init__(self,input_size:int,hidden_size:List[int],kernel_size:List[int]) -> None:
        super().__init__()
        self.hidden_size=hidden_size
        self.level=len(self.hidden_size)
        layer=[ConvLSTMCell(input_size=input_size,hidden_size=hidden_size[0],kernel_size=kernel_size[0],batch_norm=True)]
        for i in range(1,len(hidden_size)):
            layer.append(ConvLSTMCell(input_size=hidden_size[i-1],hidden_size=hidden_size[i],kernel_size=kernel_size[i],batch_norm=True))
        self.encoder=nn.ModuleList(
            layer
        )
    def forward(self,x,h:Union[List[torch.Tensor],None]=None,c:Union[List[torch.Tensor],None]=None):
        # x:(T,B,C,H,W)
        T,B,C,H,W=x.shape
        x = torch.unbind(x)

        if h is None or c is None:
            h = []
            c = []
            for i in range(self.level):
                h_t,c_t=self.init_hidden(B,self.hidden_size[i],(H,W))
                h.append(h_t)
                c.append(c_t)

        # encoder
        for t in range(T):
            h_layer=[]
            c_layer=[]
            out = x[t]
            for layer in range(len(self.encoder)):
                h_t,c_t = self.encoder[layer](out,h[layer],c[layer])
                h_layer.append(h_t)
                c_layer.append(c_t)
                out=h_t
            h,c=h_layer,c_layer
            del h_layer,c_layer
        # output (layer,B,C,H,W) 
        return h,c 
    def init_hidden(self,batch_size,hidden_size,image_size):
        '''
        return h and c
        '''
        height, width = image_size
        return (torch.zeros(batch_size, hidden_size, height, width,device=self.encoder[0].input_lin.weight.device),
                torch.zeros(batch_size, hidden_size, height, width,device=self.encoder[0].input_lin.weight.device))
class ConvLSTMSegMentation(nn.Module):
    def __init__(self,input_size:int,hidden_size:List[int],kernel_size:List[int],num_classes:int) -> None:
        super().__init__()
        self.encoder=ConvLSTM(input_size=input_size,hidden_size=hidden_size,kernel_size=kernel_size)
        self.skip_conv=nn.Sequential(
            nn.Conv2d(hidden_size[-1],input_size,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_size,input_size,kernel_size=3,stride=1,padding=1)
        )
        self.conv=nn.Sequential(
            nn.Conv2d(sum(hidden_size),num_classes,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes,num_classes,kernel_size=3,stride=1,padding=1)
        )
    def forward(self,x):
        # x:(T,B,C,H,W)
        h,c = self.encoder(x)
        out = torch.cat(h,dim=1)
        out = self.conv(out)
        return out
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ConvLSTMSegMentation(input_size=1,hidden_size=[64,128,256],kernel_size=[3,5,7],num_classes=1).to(device)
    data=torch.randn(6,4,1,160,256).to(device)
    out=model(data)
    print(out.shape)
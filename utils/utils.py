import json
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.utils.tensorboard import SummaryWriter
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
index=1
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# set random seed
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def printInfo():
    global index
    print("{}:{}M".format(index,torch.cuda.memory_allocated(0)/1e6))
    index+=1

def modelsize(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        if isinstance(m,nn.ModuleList):
            for i in range(len(m)):
                out = m[i](input_)
                if isinstance(out,tuple):
                    for data in out:
                        out_sizes.append(np.array(out.size()))
                        input_ = out
                else:    
                    out_sizes.append(np.array(out.size()))
                    input_ = out

        else:
            out = m(input_)
            out_sizes.append(np.array(out.size()))
            input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

import torch
import numpy as np

def miou(pred, target):
    pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))

    intersection = (pred == target).float()  # 计算交集，即预测的类别和真实的类别相同的像素数量
    union = pred.numel() + target.numel() - intersection.sum().float()  # 计算并集，即预测的类别和真实的类别的总像素数量

    iou = intersection.sum().float() / union  # 计算交并比
    miou = iou.mean()  # 计算平均 mIoU

    return miou.item()

class FocalLoss(nn.Module):
    '''
    Focal Loss
    '''
    def __init__(self, alpha=1, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt是预测值和真实值之间的概率
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class DiceLoss(nn.Module):
    '''
    Dice Loss
    '''
    def __init__(self) -> None:
        super().__init__()
    def forward(self,predict,label):
        # batch number
        n=predict.size()[0]
        predict=torch.sigmoid(predict)
        
        # change the shape
        predict=predict.view(n,-1)
        label=label.view(n,-1)

        # compute Dice Loss
        intersection=torch.sum(predict*label,dim=1)
        union=torch.sum(predict,dim=1)+torch.sum(label,dim=1)
        dice=(2*intersection+1e-8)/(union+1e-8)
        loss=1-torch.mean(dice)

        return loss

class PodFarCSI(nn.Module):
    '''
    POD:TP/(TP+FN)
    FAR:FP/(TP+FP)
    CSI:TP/(TP+FN+FP)  main indicator
    the general indicator about convective monitoring
    '''
    def __init__(self,gpu) -> None:
        super().__init__()
        '''   
        confusion matrix
            neg  pos
        neg
        pos
        '''
        self.matrix=torch.zeros((2,2))
        self.POD=0.0
        self.FAR=0.0
        self.CSI=0.0
        self.POD_NEG=0.0

        self.pod_num=0.0
        self.far_num=0.0
        self.csi_num=0.0
        self.pod_neg_num=0.0

    def forward(self,predict,label):

        '''
        return once precision,recall and f
        '''
        if predict.shape!=label.shape:
            raise Exception(f'predict shape {predict.shape} is different from label shape {label.shape}')
        predict=predict.detach().cpu()
        real=label.detach().cpu()

        # convert to 1 or 0
        pre=torch.sigmoid(predict)
        pre[pre > 0.5] = 1.0
        pre[pre <= 0.5] = 0.0
        
        real[real>0.5]=1.0
        real[real<=0.5]=0.0

        # caculate TP,FP,FN
        threshold=1e-8
        TP=torch.sum((torch.abs(pre-1.0)<=threshold)&(torch.abs(real-1.0)<=threshold))
        FP=torch.sum(pre)-TP
        FN=torch.sum(real)-TP
        
        TP=TP.item()
        FP=FP.item()
        FN=FN.item()

        self.matrix[1][1]+=TP
        self.matrix[1][0]+=FN
        self.matrix[0][1]+=FP

        # caculate P,R,F1
        POD=TP/(TP+FN+1e-8)
        FAR=FP/(TP+FP+1e-8)
        CSI=TP/(TP+FN+FP+1e-8)
        POD_NEG=FN/(TP+FN+1e-8)
        if POD>1e-3 and FAR>1e-3 and CSI>1e-3:
            self.POD_NEG+=POD_NEG
            self.POD+=POD
            self.FAR+=FAR
            self.CSI+=CSI
            self.pod_num+=1
            self.csi_num+=1
            self.far_num+=1
            self.pod_neg_num+=1
        return POD,FAR,CSI,POD_NEG
        
    def update(self):
        '''
        return all POD,FAR,CSI
        '''
        POD=self.POD/(self.pod_num+1e-8)
        FAR=self.FAR/(self.far_num+1e-8)
        CSI=self.CSI/(self.csi_num+1e-8)
        POD_NEG=self.POD_NEG/(self.pod_neg_num+1e-8)
        return POD,FAR,CSI,POD_NEG

class ConfusionMatrix(nn.Module):
    def __init__(self,gpu) -> None:
        super().__init__()
        '''   
        confusion matrix
            neg  pos
        neg
        pos
        '''
        self.matrix=torch.zeros((2,2),device=gpu)
        self.P=[]
        self.R=[]
        self.F1=[]
    def forward(self,predict,label):
        '''
        return once precision,recall and f
        '''
        tmp=torch.zeros((2,2))
        if predict.shape!=label.shape:
            raise Exception(f'predict shape {predict.shape} is different from label shape {label.shape}')
        
        # convert to 1 or 0
        pre=torch.sigmoid(predict)>0.5
        real=label>0.5

        # caculate TP,FP,FN
        TP=torch.sum(pre&real)
        FP=torch.sum(pre)-TP
        FN=torch.sum(real)-TP
        self.matrix[1][1]+=TP
        self.matrix[1][0]+=FN
        self.matrix[0][1]+=FP
        # caculate P,R,F1
        P=TP/(TP+FP+1e-8)
        R=TP/(TP+FN+1e-8)
        F1=(2*P*R)/(P+R+1e-8)
        if P>0.0:
            self.P.append(P.item())
        if R>0.0:
            self.R.append(R.item())
        if F1>0.0:
            self.F1.append(F1.item())
        return P,R,(2*P*R)/(P+R+1e-8)
        
    def update(self):
        '''
        return all precision,recall,f1 score
        '''
        P=self.matrix[1][1]/(self.matrix[1][1]+self.matrix[0][1]+1e-8)
        R=self.matrix[1][1]/(self.matrix[1][1]+self.matrix[1][0]+1e-8)
        return P,R,(2*P*R)/(P+R+1e-8),sum(self.P)/(len(self.P)+1e-8),sum(self.R)/(len(self.R)+1e-8),sum(self.F1)/(len(self.F1)+1e-8)
        
class PixelAccuracy(nn.Module):
    '''
    Pixel Accuracy
    '''
    def __init__(self) -> None:
        super().__init__()
    def forward(self, predict:torch.Tensor,label:torch.Tensor):
        if predict.shape!=label.shape:
            raise Exception(f'predict shape {predict.shape} is different from label shape {label.shape}')
        x1=torch.sigmoid(predict)>0.5
        x2=label>0.5
        correct=(x1==x2)
        num=1.0
        for i in correct.shape:
            num*=i
        return torch.sum(correct)/num


from typing import Union

def writeGraph(origin:torch.Tensor,label:Union[torch.Tensor,None]=None):
    toImage=ToPILImage()
    if label==None:
        if len(origin.shape)==3:
            origin=toImage(origin)
            fig=plt.figure
            plt.imshow(origin,cmap=plt.cm.gray)
            return fig
        else:
            origin=torch.unbind(origin)
            fig,ax=plt.subplots(1,len(origin))
            # plt.figure(figsize=(100,120))
            for i in range(len(origin)):
                img=toImage(origin[i])
                ax[i].imshow(img,cmap=plt.cm.gray)
                ax[i].set_title('origin')
    elif len(origin.shape)==3:
        origin=toImage(origin)
        label=toImage(torch.sigmoid(label))
        fig,(ax1,ax2)=plt.subplots(2,1)
        ax1.imshow(origin,cmap=plt.cm.gray)
        ax1.set_title('origin')
        ax2.imshow(label,cmap=plt.cm.gray)
        ax2.set_title('label')
    else:
        origin=torch.unbind(origin)
        fig,ax=plt.subplots(2,len(origin))
        # plt.figure(figsize=(100,120))
        for i in range(len(origin)):
            img=toImage(origin[i])
            labels=toImage(torch.sigmoid(label))
            ax[0][i].imshow(img,cmap=plt.cm.gray)
            ax[0][i].set_title('origin')
            ax[1][i].imshow(labels,cmap=plt.cm.gray)
            ax[1][i].set_title('label')
    plt.tight_layout()
    return fig

def drawPicture(model:nn.Module,loader,save_path:str,fig_name:str,writer:Union[SummaryWriter,None]=None):
    model.eval()
    with torch.no_grad():
        for index,batch in enumerate(loader):
            x:torch.Tensor
            y:torch.Tensor
            y_pre:torch.Tensor

            x,y=batch
            x=x.to(device)
            y=y.to(device)
            if fig_name=='unet':
                pass
            else:
                # x:(t,b,c,h,w)
                x=x.permute(1,0,2,3,4)
                # y:(t,b,c,h,w)
                y=y.permute(1,0,2,3,4)
            y_pre=model(x)
            if len(y_pre.shape)==5:
                y_pre=torch.unbind(y_pre)[-1]
                y=torch.unbind(y)[-1]
            fig=writeGraph(y.detach().cpu()[0],y_pre.detach().cpu()[0])
            fig.savefig(save_path+'/'+f'{index}.png')
            if writer:
                writer.add_figure(fig_name,fig,index+1)

def loadModel(model,path):
    new_state_dict={}
    for k, v in torch.load(str(path)).items():
        new_state_dict[k[7:]] = v			#键值包含‘module.’ 则删除 
    model.load_state_dict(new_state_dict,strict=False)
    return model

if __name__=='__main__':
    pass
    


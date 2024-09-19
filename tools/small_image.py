# different distribution of convective evaluation when 1%,2%,3%,4%,5%
import os
import sys
sys.path.append(os.getcwd())
import torch
import torchvision
import torch.nn as nn
from utils.utils import PixelAccuracy,writeGraph,drawPicture,loadModel,ConfusionMatrix,PodFarCSI,DiceLoss,miou
import argparse
from utils.datasets import CloudDataset
from torch.utils.data import DataLoader
from typing import Union
from net.models.mcsdnet import *
from utils.datasets import bulid_dataset
from config.config import load_config
from net.model import build_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose,ToTensor,Resize
from PIL import Image
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import json
valid_dataset=None
test_dataset=None
train_dataset=None
threshold=[(i/10)/100 for i in range(2,11,2)]
def distribution(cnt,total):
    for index,thres in enumerate(threshold):
        if cnt/total<=1e-8:
            return 0
        if cnt/total<thres:
            return index+1
    return 99999 
def evaluate(model:nn.Module,loader):
    print("condition evaluation!!!!!")
    metrics=PodFarCSI(gpu=device)
    result={}
    resize=torchvision.transforms.Resize((224,224))
    with torch.no_grad():
        model.eval()
        for index,batch in enumerate(loader):
            # (b,t,c,h,w)
            x,y=batch
            x=x.to(device)
            y=y.to(device)
        
            # (b,t,c,h,w)->(t,b,c,h,w)
            x=x.permute(1,0,2,3,4)
            y=y.permute(1,0,2,3,4)
            y_pre=model(x)
            # 2List
            x=torch.unbind(x)
            y=torch.unbind(y)
            y_pre=torch.unbind(y_pre)
            for j in range(len(x)):
                sample_x=x[j]
                sample_y=y[j]
                sample_y_pre=y_pre[j]
                pod,far,csi,pod_neg=metrics(sample_y_pre,sample_y)
                sample_y[sample_y>0.5]=1
                sample_y[sample_y<0.5]=0
                data=torch.where(sample_y==1)
                cnt=len(data[0])
                total=sample_y.shape[-1]*sample_y.shape[-2]
                key=distribution(cnt,total)
                # distribution statistic
                if key not in result:
                    result[key]={}
                    result[key]['POD']=pod
                    result[key]['FAR']=far
                    result[key]['CSI']=csi
                    result[key]['num']=1
                else:
                    result[key]['POD']+=pod
                    result[key]['FAR']+=far
                    result[key]['CSI']+=csi
                    result[key]['num']+=1

        pod,far,csi,pod_neg=metrics.update()
        print('POD:%.5f FAR:%.5f CSI:%.5f POD_NEG:%.5f '%(pod,far,csi,pod_neg))
        for key in result:
            result[key]['POD']=result[key]['POD']/result[key]['num']
            result[key]['FAR']=result[key]['FAR']/result[key]['num']
            result[key]['CSI']=result[key]['CSI']/result[key]['num']
        # print("=======================0%~0.2%=================================")
        # print('POD:%.5f FAR:%.5f CSI:%.5f num:%.5f'%(result[1]['POD'],result[1]['FAR'],result[1]['CSI'],result[1]['num']))
        # print("=======================0.2%~0.4%=================================")
        # print('POD:%.5f FAR:%.5f CSI:%.5f num:%.5f'%(result[2]['POD'],result[2]['FAR'],result[2]['CSI'],result[2]['num']))
        # print("=======================0.4%~0.6%=================================")
        # print('POD:%.5f FAR:%.5f CSI:%.5f num:%.5f'%(result[3]['POD'],result[3]['FAR'],result[3]['CSI'],result[3]['num']))
        # print("=======================0.6%~0.8%=================================")
        # print('POD:%.5f FAR:%.5f CSI:%.5f num:%.5f'%(result[4]['POD'],result[4]['FAR'],result[4]['CSI'],result[4]['num']))
        # print("=======================0.8%~1%=================================")
        # print('POD:%.5f FAR:%.5f CSI:%.5f num:%.5f'%(result[5]['POD'],result[5]['FAR'],result[5]['CSI'],result[5]['num']))
        return result
if __name__=='__main__':
    parse=argparse.ArgumentParser()
    args=parse.parse_args()
    # initial
    result_json={}
    config_path='/data/Jiatoka/MCSDNet/config/unetencoder_aspp.yaml'
    ckpt='/data/Jiatoka/MCSDNet/result/20240907014408_MCSDNet_f6_i30_e50_b8_50.pth'
    path='/data/Jiatoka/dataset/CRSIs'
    my_config=load_config(config_path)
    train_dataset,valid_dataset=bulid_dataset(path,config=my_config['model']['MCSDNet']['dataset'])
    config=my_config
    data_config=None
    model_config=None
    trainer_config=None

    # create dataset
    config=config['model']['MCSDNet']
    data_config=config['dataset']
    model_config=config
    series=config['dataset']['series']

    # 模型名
    model=build_model(model_config).to(device)
    model.load_state_dict(torch.load(ckpt))
    loss_fn=nn.BCEWithLogitsLoss()
    # create DataLoader
    batch=3
    train_loader=DataLoader(train_dataset,batch_size=batch)
    valid_loader=DataLoader(valid_dataset,batch_size=1)
    test_loader=DataLoader(valid_dataset,batch_size=1)
    # evaluate
    # print('train dataset evaluate')
    # evaluate(model,train_loader,loss_fn,args.name,'train',writer=writer,series=args.series)
    print('valid dataset evaluate')
    result_json=evaluate(model,valid_loader)
    with open('/data/Jiatoka/MCSDNet/tools/valid.json','w') as f:
       import json
       json.dump(result_json,f,indent=1)
    # print('valid dataset evaluate-- batch 1')
    # evaluate(model,test_loader,loss_fn,args.name,'test',writer=writer,series=args.series)

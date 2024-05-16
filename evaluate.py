import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from utils.utils import PodFarCSI,miou
import argparse
from torch.utils.data import DataLoader
from typing import Union
from utils.datasets import bulid_dataset
from config.config import load_config
from net.model import build_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose,ToTensor,Resize
from PIL import Image
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
valid_dataset=None
test_dataset=None
train_dataset=None
import logging
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',  
                    filename='./log/evaluate.log', # 日志文件名，如果没有这个参数，日志输出到console  
                    filemode='w') # 文件写入模式，“w”会覆盖之前的日志，“a”会追加到之前的日志  
def evaluate(model:nn.Module,loader,loss_fn,netname,loss_name,writer:Union[SummaryWriter,None]=None,series=False):
    metrics=PodFarCSI(gpu=device)
    with torch.no_grad():
        model.eval()
        for index,batch in enumerate(loader):
            x,y=batch
            x=x.to(device)
            y=y.to(device)
            # x:(t,b,c,h,w)
            x=x.permute(1,0,2,3,4)
            if series:
                # y:(t,b,c,h,w)
                y=y.permute(1,0,2,3,4)
            # forward
            y_pre=model(x)
            if not series:
                # (t,b,c,h,w)-->(b,c,h,w)
                y_pre=torch.unbind(y_pre)[-1]
            loss=loss_fn(y_pre,y)
            pod,far,csi,pod_neg=metrics(y_pre,y)
            if writer:
                writer.add_scalar(f"{loss_name}_loss",loss,global_step=index)
                writer.add_scalars(f'{loss_name}_PodFarCSI',{'POD':pod,'FAR':far,'CSI':csi},global_step=index)
        pod,far,csi,pod_neg=metrics.update()
        print('POD:%.5f FAR:%.5f CSI:%.5f'%(pod,far,csi))

if __name__=='__main__':
    import time
    start=time.time()
    # parse
    parse=argparse.ArgumentParser()
    parse.add_argument('-p','--path',type=str,help='model path')
    parse.add_argument('-n','--name',type=str,help='model name',default='MCSDNet')
    parse.add_argument('--config',default='./config/mcsdnet.yaml',type=str,help="config path")
    parse.add_argument('--config_name',default='MCSDNet',type=str,help="model name in config")
    parse.add_argument('--dataset',default='./data/MCSRSI',type=str,help="dataset path")

    args=parse.parse_args()
    config=args.config
    config=load_config(args.config)
    data_config=None
    model_config=None
    trainer_config=None
    # create dataset
    config=config['model'][args.config_name]
    data_config=config['dataset']
    model_config=config
    path=args.dataset
    train_dataset,test_dataset=bulid_dataset(path,config=data_config)
    args.series=config['dataset']['series']
    writer=SummaryWriter(f'./tensorboard/{args.name}')
    # create model
    try:
        model_name=args.name.split('/')[-1].split('.')[0]
    except Exception as e:
        raise e
    args.modelname=model_name
    model=build_model(config).to(device)
    model.load_state_dict(torch.load(args.path))
    loss_fn=nn.BCEWithLogitsLoss()
    test_loader=DataLoader(test_dataset,batch_size=3)
    
    # evaluate
    evaluate(model,test_loader,loss_fn,args.name,'test',writer=writer,series=args.series)
    end=time.time()
    print(f"test finished:{end-start}s")
    writer.close()

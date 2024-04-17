import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP  
import torch.distributed as dist
from datetime import datetime
from torch.nn import SyncBatchNorm as SynBN
from utils.utils import FocalLoss,PodFarCSI,DiceLoss,printInfo
from typing import Union
import os
from evaluate import evaluate
from net.scheduler import *
import sys
from torch.utils.tensorboard import SummaryWriter
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def isStep(scheduler_name):
    if 'warmup' in scheduler_name:
        return 0
    elif 'MultiStep' in scheduler_name:
        return 1
    else:
        return 2
class Trainer(object):
    '''
    a trainer
    '''
    def __init__(self,gpu:int,rank:int,world_size:int,model:nn.Module,dataset:Dataset,loss_fn:nn.Module,train_series:bool):
        self.gpu=gpu
        self.rank=rank
        self.model=model
        self.dataset=dataset
        self.loss_fn=loss_fn
        self.world_size=world_size
        self.train_series=train_series
    def __call__(self,args,batch_size,lr,epochs,netname,modelname,pin_memory=True):
        
        # load model
        device=torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() else 'cpu')
        model=self.model.to(device)
        if device.type!='cpu':
            model=model if self.world_size==1 else SynBN.convert_sync_batchnorm(model)
            model=DDP(model,device_ids=[self.gpu],find_unused_parameters=True)
        
        # load dataset
        train_sampler=DistributedSampler(self.dataset,num_replicas=self.world_size,rank=self.rank)
        train_loader=DataLoader(self.dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=pin_memory,sampler=train_sampler)
        
        # load criterion
        criterion=self.loss_fn.to(device)
        
        # create optimizer
        num_training_steps=epochs*len(train_loader)
        num_warmup_steps=num_training_steps*args.ratio
        optimizer=builid_optimizer(optimizer_name=args.optimizer,model=model,lr=lr,weight_decay=args.decay)
        scheduler=build_scheduler(scheduler_name=args.scheduler,optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,num_traning_steps=num_training_steps)
        # optimizer=torch.optim.Adam(params=model.parameters(),lr=lr)
        # scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5)
        
        # create metrics
        metrics=PodFarCSI(gpu=device)

        # create tensorboard
        if self.rank==0:
            writer=SummaryWriter(f'./tensorboard/{netname}')
        # train
        model.train()
        total_len=len(train_loader)
        mod=total_len//10+1
        st=datetime.now()
        if isStep(args.scheduler)==0:
            print('warmup')
        else:
            print('reduce')
        for epoch in range(1,epochs+1):
            train_sampler.set_epoch(epoch)
            loss_total=0.0
            num=0.0
            for index,batch in enumerate(train_loader):
                x,y=batch
                x=x.to(device)
                y=y.to(device)
                # x:(t,b,c,h,w)
                x=x.permute(1,0,2,3,4)
                if self.train_series:
                    # y:(t,b,c,h,w)
                    y=y.permute(1,0,2,3,4)
                y_pre=model(x)
                if not self.train_series:
                    # (t,b,c,h,w)-->(b,c,h,w)
                    y_pre=torch.unbind(y_pre)[-1]
                    
                loss=criterion(y_pre,y)
                loss_total+=loss.item()
                num+=1
                pod,far,csi,pod_neg=metrics(y_pre,y)
                del y,y_pre,x

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if isStep(args.scheduler)==0:
                    scheduler.step()
                if self.rank==0 and ((index+1)%(mod)==0 or index+1==total_len):
                    print('Epoch [{}/{}],Step [{}/{}],Loss:{:.4f} POD:{:.4f} FAR:{:.4f} CSI:{:.4f} POD_NEG:{:.4f}'.format(
                        epoch,
                        epochs,
                        index+1,
                        total_len,
                        loss.item(),
                        pod,
                        far,
                        csi,
                        pod_neg
                    ))
            if self.rank==0 and (epoch%args.checkpoint_num==0 or epoch==epochs):
                if not os.path.exists(f"./result/"):
                    os.makedirs(f"./result/")
                torch.save(model.module.state_dict(),f'./result/{modelname}_{epoch}.pth')
            
            if isStep(args.scheduler)==2:
                scheduler.step(loss_total/num)
            else:
                scheduler.step()
            if self.rank==0:
                writer.add_scalar('lr',optimizer.state_dict()['param_groups'][0]['lr'],epoch)
                writer.add_scalar('train_loss',loss,epoch)
                writer.add_scalars(f'train_{netname}_PodFarCSI',{'POD':pod,'FAR':far,'CSI':csi})
        if self.rank==0:
            writer.close()
            print('Training complete in:'+str(datetime.now()-st))

def train(gpu,args):
    '''
    train process
    '''
    print(f'gpu:{gpu}')
    rank=args.nr*args.gpus+gpu
    # init process group
    dist.init_process_group(
        backend='nccl',
        world_size=args.world_size,
        rank=rank
    )
    loss_fn=bulid_loss_fn(args.loss)
    trainer=Trainer(
        gpu=gpu,
        rank=rank,
        world_size=args.world_size,
        model=args.model,
        dataset=args.dataset,
        loss_fn=loss_fn,
        train_series=args.series
    )
    # start train
    trainer(batch_size=args.batch,lr=args.lr,epochs=args.epochs,pin_memory=False,netname=args.netname,modelname=args.modelname,args=args)
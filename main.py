import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.datasets import CloudDataset,bulid_dataset
from utils.utils import printInfo
from train import train
from evaluate import evaluate
from utils.utils import writeGraph,set_random_seed

import os
import sys
import argparse
import torch.multiprocessing as mp 
from torchvision.transforms import Compose,ToTensor,Resize
import datetime
from config.config import load_config 
from net.model import build_model
from torch.utils.tensorboard import SummaryWriter
SEED=0
import torch 
if __name__=='__main__':
    set_random_seed(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # set environment
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='6788'
    # parse argument
    parse=argparse.ArgumentParser('distribution setting')
    parse.add_argument('-n','--nodes',default=1,type=int,help='the number of nodes/computer')
    parse.add_argument('-g','--gpus',default=1,type=int,help='the number of gpus per nodes')
    parse.add_argument('-nr','--nr',default=0,type=int,help='ranking within the nodes')
    parse.add_argument('--modelname',default="MCSDNet",type=str,choices=["MCSDNet"],help="the model name")
    parse.add_argument('--config',default='./config/mcsdnet.yaml',type=str,help="config path")
    parse.add_argument('--config_name',default='MCSDNet',type=str,help="model name in config")
    parse.add_argument('--ckpt',default=None,type=str,help="checkpoint")
    parse.add_argument('--path',default='./data/MCSRSI',type=str,help="dataset path")
    args=parse.parse_args()
    
    # load_config
    config=args.config
    config=load_config(args.config)
    config=config['model'][args.config_name]
    data_config=config['dataset']
    model_config=config
    
    # trainer config
    args.series=config['dataset']['series']
    args.loss=config['trainer']['loss']
    args.epochs=config['trainer']['epochs']
    args.lr=config['trainer']['lr']
    args.optimizer=config['trainer']['optimizer']
    args.scheduler=config['trainer']['scheduler']
    args.ratio=config['trainer']['ratio']
    args.batch=config['dataset']['batch']
    args.ckpt=config['ckpt']
    args.checkpoint_num=config['checkpoint_num']
    args.decay=config['trainer']['decay']

    # create dataset
    path=args.path
    train_dataset,test_dataset=bulid_dataset(path,config=data_config)

    # create model
    model=build_model(model_config)
    if args.ckpt!='None':
        model.load_state_dict(torch.load(args.ckpt))

    # configure
    args.world_size=args.nodes*args.gpus
    args.dataset=train_dataset
    args.model=model
    args.netname=args.modelname
    args.frames=config['dataset']['frames']
    args.interval=config['dataset']['interval']
    args.modelname=datetime.datetime.now().strftime("%Y%m%d%H%M%S_"+args.config_name+f'_f{args.frames}_i{args.interval}_e{args.epochs}_b{args.batch}')
    print(args.modelname)

    # start train
    mp.spawn(
        train,
        nprocs=args.gpus,
        args=(args,)
    )
    
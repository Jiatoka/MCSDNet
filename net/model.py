import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import os
import sys
from config.config import load_config
import argparse
from net.module.encoder_decoder.bottom import *
from net.module.encoder_decoder.encoder import *
from net.models.mcsdnet import *
from net.module.convlstm.convlstm import *
from utils.datasets import *
from torch.utils.tensorboard import SummaryWriter
class UpSample(nn.Module):
    def __init__(self, shape,channel) -> None:
        super().__init__()
        self.shape=shape
        self.conv=nn.Conv2d(channel,1,stride=1,padding=0,kernel_size=1)
    def forward(self,x):
        # x:[level,TB,C,H,W]
        x=x[-1]
        x=self.conv(x)
        out=nn.functional.interpolate(x,size=self.shape,mode='bilinear')
        return out
def build_encoder(config,layer_channels,dataset=None):
    if config['name']=='UnetEncoder':
        if 'aspp' not in config:
            config['aspp']=False 
        encoder=UnetEncoder(config["in_channels"],layer_channels,batch=config["batch"],group_norm=config["group_norm"],interpolation=config["interpolation"],
        num_groups=config["num_groups"],residual=config["residual"],aspp=config['aspp'])
    elif config['name']=='MultiScaleEncoder':
        if 'aspp' not in config:
            config['aspp']=True
        encoder=MultiScaleEncoder(config["in_channels"],layer_channels,batch=config["batch"],group_norm=config["group_norm"],interpolation=config["interpolation"],
        num_groups=config["num_groups"],residual=config["residual"],aspp=config['aspp'])
    return encoder
def build_decoder(config,layer_channels,dataset=None):
    if config['name']=='ConvDecoder':
        decoder=ConvDecoder(
            in_layer_channels=layer_channels,
            out_layer_channels=config["out_layer_channels"],
            out_channel=config['out_channels'],
            batch=config['batch'],
            residual=config['residual'],
            group_norm=config['group_norm'],
            num_groups=config['num_groups'],
            interpolation=config['interpolation'],
            last_layer=config['last_layer'],
            level=config['level']
        )
    elif config['name']=='None':
        decoder=UpSample(shape=(dataset['height'],dataset['width']),channel=config['out_layer_channels'][-1])
    return decoder
def build_bottom(config):
    bottom=None
    if config['name']=='BottomConv':
        bottom=BottomConv(in_channels=config['in_channels'],
                        out_channels=config['out_channels'],
                        num_groups=config['num_groups'])
    elif config['name']=='BottomConv3D':
        bottom=BottomConv3D(in_channel=config['in_channels'],
                        out_channel=config['out_channels'])
    elif config['name']=='BottomViT':
        bottom=BottomViT(
            channel_in=config['in_channels'],
            channel_hid=config['out_channels'],
            N2=config['level'],
            shortcut=config['shortcut'],
            mlp_ratio=config['mlp_ratio'], 
            drop=config['drop'], 
            drop_path=config['drop_path']
        )
    elif config['name']=='None':
        bottom=nn.Identity()
    elif config['name']=='BottomConvLSTM':
        bottom=BottomConvLSTM(config['in_channels'])
    elif config['name']=='STTransformer':
        if 'pos_emb' not in config:
            config['pos_emb']=False
        if 'num_heads' not in config:
            config['num_heads']=8
        if 'normalize' not in config:
            config['normalize']='sequence'
        bottom=STTransformer(
                        channel_in=config['in_channels'],
            channel_hid=config['out_channels'],
            N2=config['level'],
            kernel_size=config['kernel_size'],
            shortcut=config['shortcut'],
            mlp_ratio=config['mlp_ratio'], 
            drop=config['drop'], 
            drop_path=config['drop_path'],
            attn_shortcut=config['attn_shortcut'],
            dynamic=config['dynamic'],
            fuse=config['fuse'],
            dilation=config['dilation'],
            reduction=config['reduction'],
            timesteps=config['timesteps'],
            conv=config['conv'],
            spatio_attn=config['spatio_attn'],
            temporal_attn=config['temporal_attn'],
            groups=config['groups'],
            aggregation=config['aggregation'],
            shape=config['shape'],
            temporal=config['temporal'],
            spatial=config['spatial'],
            mlp_layer=config['mlp_layer'],
            pos_emb=config['pos_emb'],
            num_heads=config['num_heads'],
            normalize=config['normalize']
        )
    return bottom
def build_model(config:None):
    layer_channels=config['layer_channels']
    encoder=build_encoder(config['encoder'],layer_channels)
    decoder=build_decoder(config['decoder'],layer_channels,config['dataset'])
    bottom=build_bottom(config=config['bottom'])
    model=MCSDNet(encoder=encoder,decoder=decoder,bottom=bottom)
    return model
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parse=argparse.ArgumentParser()
    parse.add_argument('--modelname',default='MCSDNet')
    args=parse.parse_args()
    config=load_config("./config/longmcsdnet15.yaml")
    config=config['model']['MCSDNet']
    data_config=config['dataset']
    path='/data/Jiatoka/dataset/CRSIs'
    train_dataset,test_dataset=bulid_dataset(path,config=data_config)
    print(train_dataset[0][0].shape)
    model=build_model(config=config).to(device)
    data=torch.randn(12,4,1,160,256).to(device)
    out=model(data)
    print(out.shape)
    # print(model)
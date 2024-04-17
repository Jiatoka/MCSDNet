# generate MCS detection result image
from torch import nn
import torch
import argparse
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.transforms import ToPILImage,Resize,Compose,ToTensor
from matplotlib.pyplot import plot as plt
import os
from utils.datasets import bulid_dataset
from config.config import load_config
from net.model import build_model
from torch.utils.data import DataLoader
from utils.datasets import bulid_dataset,CloudDataset
def writeGraph(origin:torch.Tensor,label:torch.Tensor,predict:torch.tensor,image_name:str):
    image_name=image_name.split('.')[0]
    # origin:(c,h,w)
    # label:(c,h,w)
    # predict:(c,h,w)

    # label
    label[label > 0.5] = 1.0
    label[label <= 0.5] = 0.0

    
    # predict 
    predict=torch.sigmoid(predict)
    predict[predict > 0.5] = 1.0
    predict[predict <= 0.5] = 0.0
    

    # tensor to image
    toImage=ToPILImage()
    origin=toImage(origin)
    label=toImage(label)
    predict=toImage(predict)
    if not os.path.exists(f"./image/origin"):
        os.makedirs(f"./image/origin")
        os.makedirs(f"./image/label")
        os.makedirs(f"./image/monitor")
    image_name=image_name[0:12]
    origin.save(f"./image/origin/{image_name}.png")
    label.save(f"./image/label/{image_name}.png")
    predict.save(f"./image/monitor/{image_name}.png")
if __name__=='__main__':
    parse=argparse.ArgumentParser()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parse.add_argument('--path',type=str,help='the path of dataset')
    parse.add_argument('--checkpoint',type=str,help='the checkpoint of MCSDNet')
    args=parse.parse_args()
    config=load_config('./config/mcsdnet.yaml')
    config=config['model']['MCSDNet']
    data_config=config['dataset']
    model=build_model(config=config).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    path=args.path
    data_config['frames']=6
    data_config['interval']=15
    data_config['series']=True
    data_config['batch']=1
    train_dataset,test_dataset=bulid_dataset(path,config=data_config)
    dataset=test_dataset
    with torch.no_grad():
        model.eval()
        # valid dataset
        loader=DataLoader(dataset,batch_size=1)
        for index,data in enumerate(loader):
            x,y=data
            
            # (b,t,c,h,w) 
            input_data=x.to(device)
            label=y.to(device)
            # (t,b,c,h,w)
            input_data=input_data.permute(1,0,2,3,4)
            label=label.permute(1,0,2,3,4)
            
            # (t,b,c,h,w)
            output=model(input_data)
            # reshape
            # (t,c,h,w)
            input_data=input_data.permute(1,0,2,3,4)
            input_data=torch.unbind(input_data)[-1]
            label=label.permute(1,0,2,3,4)
            label=torch.unbind(label)[-1]
            output=output.permute(1,0,2,3,4)
            output=torch.unbind(output)[-1]
            for t in range(output.shape[0]):
                writeGraph(origin=input_data[t],label=label[t],
                predict=output[t],image_name=dataset.getName(index)[t])
            print(f"=====image {dataset.getName(index)[0]} predict finished=====")
        
        # train dataset
        loader=DataLoader(train_dataset,batch_size=1)
        for index,data in enumerate(loader):
            x,y=data
            
            # (b,t,c,h,w) 
            input_data=x.to(device)
            label=y.to(device)
            # (t,b,c,h,w)
            input_data=input_data.permute(1,0,2,3,4)
            label=label.permute(1,0,2,3,4)
            
            # (t,b,c,h,w)
            output=model(input_data)
            # reshape
            # (t,c,h,w)
            input_data=input_data.permute(1,0,2,3,4)
            input_data=torch.unbind(input_data)[-1]
            label=label.permute(1,0,2,3,4)
            label=torch.unbind(label)[-1]
            output=output.permute(1,0,2,3,4)
            output=torch.unbind(output)[-1]
            for t in range(output.shape[0]):
                writeGraph(origin=input_data[t],label=label[t],
                predict=output[t],image_name=train_dataset.getName(index)[t])
            print(f"=====image {train_dataset.getName(index)[0]} predict=====")
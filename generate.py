# generate MCS detection result image
import os
import sys
sys.path.append(os.getcwd())
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
    # predict:(c,h,w)
    
    # predict 
    predict=torch.sigmoid(predict)
    predict[predict > 0.5] = 1.0
    predict[predict <= 0.5] = 0.0

    # tensor to image
    toImage=ToPILImage()
    origin=toImage(origin)
    predict=toImage(predict)
    if not os.path.exists(f"./image/origin"):
        os.makedirs(f"./image/origin")
        os.makedirs(f"./image/monitor")
    image_name=image_name[0:12]
    origin.save(f"./image/origin/{image_name}.png")
    predict.save(f"./image/monitor/{image_name}.png")
if __name__=='__main__':
    parse=argparse.ArgumentParser()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parse.add_argument('--checkpoint',type=str,default='./result/20240402190353_MCSDNet_f6_i30_best.pth')
    parse.add_argument('--imageDir',type=str,default='./example',help="the input image sequence which aims to detect convective cloud")
    args=parse.parse_args()

    # create model
    config=load_config('./config/mcsdnet.yaml')
    config=config['model']['MCSDNet']
    model=build_model(config=config).to(device)
    model.load_state_dict(torch.load(args.checkpoint))

    # generate images
    with torch.no_grad():
        model.eval()

        from pathlib import Path
        image_name_list=os.listdir(args.imageDir)
        image_name_list=sorted(image_name_list)
        for i in range(len(image_name_list)-5):
            # open image
            image_names=image_name_list[i:i+6]
            images=[]
            for img in image_names:
                img_path=os.path.join(args.imageDir,img)
                from PIL import Image
                data=Image.open(img_path)
                data=data.convert('L')
                transforms=Compose([Resize([160,256]),ToTensor()])
                data=transforms(data)
                images.append(data)
            images=torch.stack(images)
            images=images.unsqueeze(dim=0)
            images=images.to(device)

            # inference
            # (t,b,c,h,w)
            images=images.permute(1,0,2,3,4)
            # (t,b,c,h,w)
            output=model(images)
            
            # reshape
            # (b,t,c,h,w)
            images=images.permute(1,0,2,3,4)
            images=torch.unbind(images)[-1]
            output=output.permute(1,0,2,3,4)
            output=torch.unbind(output)[-1]

            # draw images
            for t in range(output.shape[0]):
                writeGraph(origin=images[t],label=None,
                predict=output[t],image_name=image_names[t])
            print(f"=====Image:{image_names[0]} detection finish=====")

        
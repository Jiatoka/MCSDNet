# generate MCS detection result image
import os
import sys
sys.path.append(os.getcwd())
from torch import nn
import torch
import argparse
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision.transforms import ToPILImage,Resize,Compose,ToTensor
import matplotlib.pyplot as plt
import os
from utils.datasets import bulid_dataset
from config.config import load_config
from net.model import build_model
from torch.utils.data import DataLoader
from utils.datasets import bulid_dataset,CloudDataset
import seaborn as sns
import numpy as np
from torchvision.transforms import Compose,ToTensor,Resize
def heatmap(data,figname):
    resize=Compose([Resize((160,256))])
    data=resize(data) 
    # 选择要绘制的通道，例如第一个通道 (index 0)
    channel_data = data[0].cpu().numpy()  # 转换为 numpy 数组
    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(8,5))
    sns.heatmap(channel_data, cmap='viridis',cbar=False, xticklabels=False, yticklabels=False)  # 选择合适的颜色映射
    # 去掉白色背景
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 保存图像，紧凑地保存去除所有边距
    plt.savefig(figname, bbox_inches='tight', pad_inches=0)
    plt.savefig(figname)
    plt.close()
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
def get_feature_maps(model,args):
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
                T,B,C,H,W=images.shape
                # get heatmap
                # (tb,c,h,w)
                multi_scale_feature_maps,encoder_feature_maps,align_maps=model.get_feature_maps(images)
                print(multi_scale_feature_maps.shape)
                print(len(encoder_feature_maps))
                print(len(align_maps))
                heatmap(multi_scale_feature_maps[0],"./feature_maps.png")
                heatmap(encoder_feature_maps[0][0],"./feature_maps_level0.png")
                for i in range(len(align_maps)):
                    heatmap(align_maps[i][0],f"./align{i}.png")



                # reshape
                # (b,t,c,h,w)
                # images=images.permute(1,0,2,3,4)
                # images=torch.unbind(images)[-1]
                # output=output.permute(1,0,2,3,4)
                # output=torch.unbind(output)[-1]

                # # draw images
                # for t in range(output.shape[0]):
                #     writeGraph(origin=images[t],label=None,
                #     predict=output[t],image_name=image_names[t])
                # print(f"=====Image:{image_names[0]} detection finish=====") 
if __name__=='__main__':
    parse=argparse.ArgumentParser()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parse.add_argument('--path',type=str,help='the path of dataset')
    parse.add_argument('--checkpoint',type=str,default='./result/20240402190353_MCSDNet_f6_i30_best.pth')
    parse.add_argument('--imageDir',type=str,default='./example',help="the input image sequence which aims to detect convective cloud")
    args=parse.parse_args()

    # create model
    config=load_config('./mcsdnet.yaml')
    config=config['model']['MCSDNet']
    model=build_model(config=config).to(device)
    model.load_state_dict(torch.load(args.checkpoint))

    # heatmap
    get_feature_maps(model,args)
    